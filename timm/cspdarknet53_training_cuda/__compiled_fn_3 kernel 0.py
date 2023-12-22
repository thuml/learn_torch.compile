
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7df2wzqzj65kpbrw3kvwd46dfszadkdpcxhoozudelgek5bfog5.py
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
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ip/cipbfxel4amd5pqfoaslwrdgqq7tqyro4nlv54y5bvfxu4qihtga.py
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
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rf/crfbsjdbm7yqybgnofhsgcvb2gomsew6prpbri2qowvbace5f5gb.py
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
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxxn5napgbjqhvjn2f4dcq56ej6ulkhuyatecs2tyodzqc2sc7t.py
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
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yb/cybn3zyxxsw7ildnz4bxcbenl4i2qduha3pumr652d5tjahs2egp.py
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
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hy/chyn4idjzjo5zzd5q4dpanwd7xojpgukyl4ptlewawaxpqtrj2wz.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5e/c5ec2gq3ymq545ycql7kk57alufxd7ufyn2eb7vds2lt3tokztik.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ei/cei2e2mbqhwc6i33g2jx6crf63bkheimsworbyakt3hbbbq5dijb.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26wgbbm5oc5ctw34oihp7vlin35qhi54tjtzjmi5gy6tazjg42m.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mgt3so5enan6semhrc3gpsvvxqhj27htdvwcsbyr4y3czrms7f.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tw/ctw77uccb44fcbn4yxg5rbqqhxoeurf7hizyf7suyqhmknp5ueet.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnyb4l4i3rtotliefvio6s64nchgljegy3s5txybjpdzlp4srzn.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (2097152*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2h3dbk34w6kuwglk7kbp2t2jkrthz77vv5r66qs35wur6zkxd2.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplpfg2z6xib2wgdkjtx4tkgakkasymezulcfkniofvsuzmm74dn.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwl55qurhar4xx2elz5kgwkc53g2l3mc73vcsd26mfnj3p2rrnt.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 524288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000019073522708
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpdur2akhe2klpqd2hlijlwqdh255qorcyf5xda75sa4r5ixqwe.py
# Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_5 => gt, mul_7, where
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 524288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6voowd54pjs3ofiaqbysy5ot5kalx34smisglly556nwoopckff.py
# Source Nodes: [x_6], Original ATen: [aten.convolution]
# x_6 => convolution_1
triton_poi_fused_convolution_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjre3lhxzpubpmjzdg7yayfxztipqigdhotuwp4hrksergyyfdcy.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyshdxlebr344oi6kc65dmxkvc67vayzrdq3uyzrfb5chl3ltn3.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrbpbxieatywu5au7vurfg35kjeuu667k6qxgsr36tc2yfuaxkt.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_13, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjxvpweemqyw7ijwczjjw4aejjanra4ehj3l4zpdg4dhn2bgsaf.py
# Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_10 => gt_1, mul_15, where_1
# x_7 => add_6, add_9, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lzyl2xqnecc3wa7h53ovmhw6pdjqwfxubrbh6e3jrcicqwkm55.py
# Source Nodes: [x_13], Original ATen: [aten.convolution]
# x_13 => convolution_2
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (2097152*y1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmulleac2aedbt4lbyafhbqjhzpm2bmqgxkwevdkgbaas2yx4lij.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xmgsurkjgddj2a3vbtxgkhcukvguva4jwzb5vlbueq7fp52fft.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjanasvfqyvhfivpt55keijobgdqcwjac2wdrc52222xhhekjsa.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => add_11, add_12, add_13, mul_17, mul_18, mul_19, mul_20, mul_21, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7e3ctxpck2hbhfi2u6e4zdaxbplvwjorqwwda37gocoascije5m.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => add_11, add_14, mul_16, mul_22, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53jw46i6tjbnop62wjmatxqxc6xjld5ehrzgbvpt4ttyqz2igtl.py
# Source Nodes: [x_18], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# x_18 => gt_2, mul_23, where_2
triton_poi_fused_leaky_relu_leaky_relu_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp5, None)
    tl.store(out_ptr1 + (y0 + (128*x2) + (2097152*y1)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/cco5ajb3m5chvu7nlqrzf6alytgtevt5337vyfgf6nb6eb3jezqm.py
# Source Nodes: [x_19], Original ATen: [aten.convolution]
# x_19 => convolution_3
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (1048576 + x2 + (16384*y0) + (2097152*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdayquezgkl4hhvudohdrhjfbpmjt7bl77eamc2pdmb2bh3huvug.py
# Source Nodes: [x_19], Original ATen: [aten.convolution]
# x_19 => convolution_3
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (524288*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2ga6hhgx23wyn3j6zhefpng2zudztzmd56np7e4hsm7ropbhky2.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebt62wg4swgkfkdi2bpbjprie6aiva4b7j2iirtvfuuyoxztmfl.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => add_16, add_17, add_18, mul_25, mul_26, mul_27, mul_28, mul_29, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7a4iy6dyujlryca37sxgyvcqsoutre6c3g5ruyqoqr3rtpc4pd.py
# Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_20 => add_16, add_19, mul_24, mul_30, rsqrt_3, sub_3, var_mean_3
# x_24 => gt_3, mul_31, where_3
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjj2dspb3gflfr7f5xahv5gv2fvzqmynxumdbaqxv2u2uz55yrb5.py
# Source Nodes: [x_27, x_31, xb_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# x_27 => add_21, add_24, mul_32, mul_38, rsqrt_4, sub_4, var_mean_4
# x_31 => gt_4, mul_39, where_4
# xb_1 => add_25
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 16384
    y3 = (yindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x1 + (64*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (1048576 + y2 + (16384*x1) + (2097152*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x1 + (64*y0)), tmp20, xmask)
    tl.store(out_ptr2 + (x1 + (64*y0)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyouspvi52yteylkc65jjt6n74ywfjf7jnieahweqth524pwak5.py
# Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_34 => add_27, add_30, mul_40, mul_46, rsqrt_5, sub_5, var_mean_5
# x_37 => gt_5, mul_47, where_5
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cli62webgjforlpelsenvl4of6z6wdhmxyqawzxhnwyqcjxcchom.py
# Source Nodes: [cat_9], Original ATen: [aten.cat]
# cat_9 => cat
triton_poi_fused_cat_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16384
    y1 = (yindex // 16384)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (16384*x2) + (2097152*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-64) + x2 + (64*y3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxb5rlmtycqnovchreiwz27s4oepjxxfdbb53hg36ir23odhqhf.py
# Source Nodes: [x_43], Original ATen: [aten.convolution]
# x_43 => convolution_7
triton_poi_fused_convolution_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2aimon2nst2rw2d2kewytyqbgcs4euvcxnef72f5uypjicssxf.py
# Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
# x_44 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejtodx2soyspspyc7xjpoucw5aajuuhzaaeyap36pbt5lewom22.py
# Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
# x_44 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chuoxus5bhs3afqgjafksmekitwuktqefwbofm7fukfobai36zxy.py
# Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
# x_44 => add_37, add_38, add_39, mul_57, mul_58, mul_59, mul_60, mul_61, rsqrt_7, squeeze_22, var_mean_7
triton_per_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwx2un4tgxrinyjjto6be34oe7b3oxp7hf5k2k5bpyv2zuclbn5d.py
# Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_44 => add_37, add_40, mul_56, mul_62, rsqrt_7, sub_7, var_mean_7
# x_47 => gt_7, mul_63, where_7
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ywaarlakmyz7bioxmrqar2qwawqnq7xjzxhdw7mntiifopnu6p.py
# Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
# x_51 => add_42, add_45, mul_64, mul_70, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqfya5aoj4c6vgqoeuae6wjip7e2fqrcq5mcj3czcbmj66yv4kp.py
# Source Nodes: [x_55], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# x_55 => gt_8, mul_71, where_8
triton_poi_fused_leaky_relu_leaky_relu_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (524288*y1)), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x2 + (4096*y3)), tmp5, None)
    tl.store(out_ptr1 + (y0 + (128*x2) + (524288*y1)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5rhzckgefhl3n3ulzjewcbbllmvl4ovigxeap3ieg22libghwx.py
# Source Nodes: [x_56], Original ATen: [aten.convolution]
# x_56 => convolution_9
triton_poi_fused_convolution_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (262144 + x2 + (4096*y0) + (524288*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hh/chh2otiytgs3cjev4fy6jq3fihzokcxejhwzgbbpsej3nmuaxkzy.py
# Source Nodes: [x_56], Original ATen: [aten.convolution]
# x_56 => convolution_9
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvakgmudvipdo3327aninrtgfzehhwt6rgvp7srz5mjdam2fkbl5.py
# Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
# x_57 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gxiijftzrngzbrdlgdxtqdagyu3w6a62skvelva4rvug2hfibm.py
# Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
# x_57 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pt/cptxar4bh5un54cqll3vkhyhxiidfrpdflshkahjk5zq575khue4.py
# Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
# x_57 => add_47, add_48, add_49, mul_73, mul_74, mul_75, mul_76, mul_77, rsqrt_9, squeeze_28, var_mean_9
triton_per_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvplko2u7yfrhxxggnmh5mlhy2lveo3ik3ugrxi7wei23avzsleu.py
# Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_57 => add_47, add_50, mul_72, mul_78, rsqrt_9, sub_9, var_mean_9
# x_61 => gt_9, mul_79, where_9
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnhksmnal5az54n4444qt5dzpenfgt7pvwebjfqteippaht24oq.py
# Source Nodes: [shortcut_2, x_64, x_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_2 => add_56
# x_64 => add_52, add_55, mul_80, mul_86, rsqrt_10, sub_10, var_mean_10
# x_68 => gt_10, mul_87, where_10
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 4096
    y3 = (yindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x1 + (64*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (262144 + y2 + (4096*x1) + (524288*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x1 + (64*y0)), tmp20, xmask)
    tl.store(out_ptr2 + (x1 + (64*y0)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7srs5oy3stspfbqeozgr4c7nexxy5tsvqcw5pcqsvlfksg2rt7.py
# Source Nodes: [x_78, x_82, xb_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# x_78 => add_63, add_66, mul_102, mul_96, rsqrt_12, sub_12, var_mean_12
# x_82 => gt_12, mul_103, where_12
# xb_4 => add_67
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7a4zmbyi2tp3x6uwexj2yfel6a5sn7awqqymlldxga35iqgpoj.py
# Source Nodes: [x_85, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_85 => add_69, add_72, mul_104, mul_110, rsqrt_13, sub_13, var_mean_13
# x_88 => gt_13, mul_111, where_13
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4sb2clqm6duvb7uja75cbighljytwqjouvo6n566yeghhwbvta.py
# Source Nodes: [cat_8], Original ATen: [aten.cat]
# cat_8 => cat_1
triton_poi_fused_cat_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (4096*x2) + (524288*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-64) + x2 + (64*y3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ufvsfg6bc3xjbkdagxof2iclh2zxvo7hygfbrvlouxkvlyksu5.py
# Source Nodes: [x_94], Original ATen: [aten.convolution]
# x_94 => convolution_15
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (262144*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zj6f32kxwk6tfnvvwl557qznawt5xgp776xfq2fnewt4imocfr.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => var_mean_15
triton_red_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtke7a4jxhy7gstjwijyzjdr5ifcj72rgt7zke4sx2u75qmxeiy.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => add_79, add_80, add_81, mul_121, mul_122, mul_123, mul_124, mul_125, rsqrt_15, squeeze_46, var_mean_15
triton_per_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnj7p4yapfwqi3xnxqf7gssvk4bwoxyis4yzebvxjfgodjjtttyq.py
# Source Nodes: [x_95, x_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_95 => add_79, add_82, mul_120, mul_126, rsqrt_15, sub_15, var_mean_15
# x_98 => gt_15, mul_127, where_15
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgos5texzyq5w6x3zzp2xhe3fvcia3tfir4eohcj3ql3cqgrgtcw.py
# Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
# x_102 => add_84, add_87, mul_128, mul_134, rsqrt_16, sub_16, var_mean_16
triton_poi_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4q7tlbf2hmol7wjq6mb6ymebmxxpzolrtk23w7l43t5canbywl3.py
# Source Nodes: [x_106], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# x_106 => gt_16, mul_135, where_16
triton_poi_fused_leaky_relu_leaky_relu_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp5, xmask)
    tl.store(out_ptr1 + (y0 + (256*x2) + (262144*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5h5mipdt7a2xpcc7mxeuclq3ff5wgvqosiixv3zv6vmkoo6vgh.py
# Source Nodes: [x_107], Original ATen: [aten.convolution]
# x_107 => convolution_17
triton_poi_fused_convolution_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (131072 + x2 + (1024*y0) + (262144*y1)), xmask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nzvxydmv5ozm4zsdqz2yverwq2ljdhtwvok6kasyqrsg2n3p2z.py
# Source Nodes: [x_107], Original ATen: [aten.convolution]
# x_107 => convolution_17
triton_poi_fused_convolution_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqeksghjmafmliywikotft5jmmxzvfeqdd6af2fzsdepnbgfz6f.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
# x_108 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfeeebaslhbl5vbrlp5byb2wa63l55jv232qk3xiit5c7st5sbrs.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
# x_108 => add_89, add_90, add_91, mul_137, mul_138, mul_139, mul_140, mul_141, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbpfdluapo64hcrzkcpdgw4zqs5tyucyhdrniz5rffthdrtyaj4.py
# Source Nodes: [x_108, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_108 => add_89, add_92, mul_136, mul_142, rsqrt_17, sub_17, var_mean_17
# x_112 => gt_17, mul_143, where_17
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvcflcv6det5mrxecvdq3xvsobhuxkuir6wve5zomvwnh7q3jz4.py
# Source Nodes: [shortcut_4, x_115, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_4 => add_98
# x_115 => add_94, add_97, mul_144, mul_150, rsqrt_18, sub_18, var_mean_18
# x_119 => gt_18, mul_151, where_18
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 1024
    y3 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x1 + (128*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (131072 + y2 + (1024*x1) + (262144*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x1 + (128*y0)), tmp20, xmask)
    tl.store(out_ptr2 + (x1 + (128*y0)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54soua2eych4g2upqncvfkwezgyq4yhptz4fvnyox4aq24zx4p7.py
# Source Nodes: [shortcut_5, x_129, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_5 => add_109
# x_129 => add_105, add_108, mul_160, mul_166, rsqrt_20, sub_20, var_mean_20
# x_133 => gt_20, mul_167, where_20
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcodbiqit62hzulqypkekipoixbuna32wzvjcbdnxvohslprod2.py
# Source Nodes: [x_220, x_223], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_220 => add_177, add_180, mul_264, mul_270, rsqrt_33, sub_33, var_mean_33
# x_223 => gt_33, mul_271, where_33
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3xi7s32sw4pnwkqkulw2hlvubfqvzpwciqjaz35xg2wqgs5jvs.py
# Source Nodes: [cat_7], Original ATen: [aten.cat]
# cat_7 => cat_2
triton_poi_fused_cat_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (1024*x2) + (262144*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 256, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-128) + x2 + (128*y3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevtsnc2bpaytbm5lgziwwnkzaaqec2oo5wjgtac6pfjpkgphzqm.py
# Source Nodes: [x_229], Original ATen: [aten.convolution]
# x_229 => convolution_35
triton_poi_fused_convolution_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokktw3ge7hchnvf763kmjgm5nisicxaboswrqtsbmrd6ain7fph.py
# Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
# x_230 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpuasd4plrerilbhhe6ehlph6dqpnvduppuj6qyemoxp7pegkpwv.py
# Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
# x_230 => add_187, add_188, add_189, mul_281, mul_282, mul_283, mul_284, mul_285, rsqrt_35, squeeze_106, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_69', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkala3bdoscdu6yucyos5kmifdqmvivd37am7ivu626kemlcovm.py
# Source Nodes: [x_230, x_233], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_230 => add_187, add_190, mul_280, mul_286, rsqrt_35, sub_35, var_mean_35
# x_233 => gt_35, mul_287, where_35
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_70', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirbuf33vfo33h6u3o2tlmk3dnntzsyogev7hshhcnf7mjevot2s.py
# Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_functional]
# x_237 => add_192, add_195, mul_288, mul_294, rsqrt_36, sub_36, var_mean_36
triton_poi_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxhsvd5migczov4gdqisgvvbagdefuuhrxfpps3uevma62jkqaq.py
# Source Nodes: [x_241], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# x_241 => gt_36, mul_295, where_36
triton_poi_fused_leaky_relu_leaky_relu_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp5, xmask)
    tl.store(out_ptr1 + (y0 + (512*x2) + (131072*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s6gnjjogdolcahlkjx4w5wyv2k3u5uj74x6sgqs56cqp7q3abl.py
# Source Nodes: [x_242], Original ATen: [aten.convolution]
# x_242 => convolution_37
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (65536 + x2 + (256*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhvvowzeoasahh6mtbegmv6jjr5hlk6myyl6ukdxu66w6np5ono.py
# Source Nodes: [x_242], Original ATen: [aten.convolution]
# x_242 => convolution_37
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmr4aj3skcl64yq6hdbrlefateks4wltib5mbvt7edsdcpudoez6.py
# Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
# x_243 => var_mean_37
triton_red_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pbm7cyo5topill2xbk2c75pxjhwvu3bqq4trg3i632xmehbddi.py
# Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
# x_243 => add_197, add_198, add_199, mul_297, mul_298, mul_299, mul_300, mul_301, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4ktfx7jsdw76io7hqd2fnwq4t5abno37yjz3x5sdsyt5gh7xsh6.py
# Source Nodes: [x_243, x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_243 => add_197, add_200, mul_296, mul_302, rsqrt_37, sub_37, var_mean_37
# x_247 => gt_37, mul_303, where_37
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/che54zvv3na5gq2xviaftcjzbijshwahleca6gs5lqo6sxlxgfpb.py
# Source Nodes: [shortcut_12, x_250, x_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_12 => add_206
# x_250 => add_202, add_205, mul_304, mul_310, rsqrt_38, sub_38, var_mean_38
# x_254 => gt_38, mul_311, where_38
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 256
    y3 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x1 + (256*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (65536 + y2 + (256*x1) + (131072*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x1 + (256*y0)), tmp20, xmask)
    tl.store(out_ptr2 + (x1 + (256*y0)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gi7isxfja6htewt45ssetgy4tacct3fb5cxtxc6eppvzctorlf.py
# Source Nodes: [shortcut_13, x_264, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_13 => add_217
# x_264 => add_213, add_216, mul_320, mul_326, rsqrt_40, sub_40, var_mean_40
# x_268 => gt_40, mul_327, where_40
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yzthm7xqoswjueuhmnkcgfsgj4gyy2x3x52busps4tjobus7dk.py
# Source Nodes: [x_355, x_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_355 => add_285, add_288, mul_424, mul_430, rsqrt_53, sub_53, var_mean_53
# x_358 => gt_53, mul_431, where_53
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4pfy5kxl77f5mejyxyhvx55mkyysputsfxtkhi6kzf5h2nj27h.py
# Source Nodes: [cat_6], Original ATen: [aten.cat]
# cat_6 => cat_3
triton_poi_fused_cat_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (256*x2) + (131072*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 512, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-256) + x2 + (256*y3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxs3k4cyrha76x72qllrq25dyuwrg5kgouse5r2dl2z3rkih2ydo.py
# Source Nodes: [x_364], Original ATen: [aten.convolution]
# x_364 => convolution_55
triton_poi_fused_convolution_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (65536*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuiodnhrbjpd362a4qsboojnrqjupsinydu3qvln6x4zddcmz26.py
# Source Nodes: [x_365], Original ATen: [aten._native_batch_norm_legit_functional]
# x_365 => var_mean_55
triton_red_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktt6a3wccxp2nc23hh24f7q4k76cbkdkfzvxy6n4nlz5znbez4b.py
# Source Nodes: [x_365], Original ATen: [aten._native_batch_norm_legit_functional]
# x_365 => add_295, add_296, add_297, mul_441, mul_442, mul_443, mul_444, mul_445, rsqrt_55, squeeze_166, var_mean_55
triton_per_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0019569471624266
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7i/c7iejqpxxjbkjtfhcqtlm7mtbj7zzrt4hkeay4oaesbgmffo2jvo.py
# Source Nodes: [x_365, x_368], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_365 => add_295, add_298, mul_440, mul_446, rsqrt_55, sub_55, var_mean_55
# x_368 => gt_55, mul_447, where_55
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6inkjxsbspcxbu226knhbpgala6dusfghqymfuo5gxvn2m4f23p.py
# Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
# x_372 => add_300, add_303, mul_448, mul_454, rsqrt_56, sub_56, var_mean_56
triton_poi_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl433em43xpp6o4o6eyqnxtd4lw3nonecxbaco6i4ww22wcyoar5.py
# Source Nodes: [x_376], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# x_376 => gt_56, mul_455, where_56
triton_poi_fused_leaky_relu_leaky_relu_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp5, xmask)
    tl.store(out_ptr1 + (y0 + (1024*x2) + (65536*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa4u77ipfcfudbam2xf27mh4maumsa4jry4phjhupcn6ih6sgd7.py
# Source Nodes: [x_377], Original ATen: [aten.convolution]
# x_377 => convolution_57
triton_poi_fused_convolution_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (32768 + x2 + (64*y0) + (65536*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvz6lxl5zeljzlufcalt5bbgbqhqnniogh6kpx4l5ildyx4ntmgg.py
# Source Nodes: [x_377], Original ATen: [aten.convolution]
# x_377 => convolution_57
triton_poi_fused_convolution_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jho5hiix7hio4nbd65yumhqivdaezpjvh2ltrohxhdq3e6zhwc.py
# Source Nodes: [x_378], Original ATen: [aten._native_batch_norm_legit_functional]
# x_378 => var_mean_57
triton_red_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cerrh37hvhtrhogf3q2kkutoo3o3k6vaqrdtthfxleso5tqprn47.py
# Source Nodes: [x_378], Original ATen: [aten._native_batch_norm_legit_functional]
# x_378 => add_305, add_306, add_307, mul_457, mul_458, mul_459, mul_460, mul_461, rsqrt_57, squeeze_172, var_mean_57
triton_per_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0019569471624266
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp26ehgsrukw5r3pvietb3sugxver7pf4ba45dvqpcmyf2okydkb.py
# Source Nodes: [x_378, x_382], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
# x_378 => add_305, add_308, mul_456, mul_462, rsqrt_57, sub_57, var_mean_57
# x_382 => gt_57, mul_463, where_57
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwb4jgkxbv3xftcm2rurbydf7j4aaifgwrejtgvag7bnym2gmzhs.py
# Source Nodes: [shortcut_20, x_385, x_389], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_20 => add_314
# x_385 => add_310, add_313, mul_464, mul_470, rsqrt_58, sub_58, var_mean_58
# x_389 => gt_58, mul_471, where_58
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 64
    y3 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x1 + (512*y0)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (32768 + y2 + (64*x1) + (65536*y3)), xmask & ymask)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x1 + (512*y0)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (x1 + (512*y0)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czl7k7ivhaxipnuqnuzin22u5mc4w75hrr5ml4qtk7einwdmgzvw.py
# Source Nodes: [shortcut_21, x_399, x_403], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
# shortcut_21 => add_325
# x_399 => add_321, add_324, mul_480, mul_486, rsqrt_60, sub_60, var_mean_60
# x_403 => gt_60, mul_487, where_60
triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 > tmp14
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csag4h4h365d2em7i2mxxj6ptrhqkfpabufu3ss6s5btyi26fqvu.py
# Source Nodes: [x_434, x_437], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_434 => add_349, add_352, mul_520, mul_526, rsqrt_65, sub_65, var_mean_65
# x_437 => gt_65, mul_527, where_65
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bo/cbojkfbg4kvzle5cgzhhqchoqh6qi7p36igvsatb36jasm3v446r.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat_4
triton_poi_fused_cat_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (64*x2) + (65536*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1024, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-512) + x2 + (512*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlc2r4u34vhc3ddwqw5jcqxuwh5fz3yneragyaufc5vgen6cd2l.py
# Source Nodes: [x_439, x_444], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
# x_439 => add_354, add_357, mul_528, mul_534, rsqrt_66, sub_66, var_mean_66
# x_444 => gt_66, mul_535, where_66
triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.01
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tmp18 > tmp14
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvmqe4p55eclwlnoua4t64if2rtgtkgyl4cmu4u3n6k7uhhujtn.py
# Source Nodes: [x_444, x_445, x_447], Original ATen: [aten.leaky_relu, aten.mean, aten.view]
# x_444 => gt_66, mul_535, where_66
# x_445 => mean
# x_447 => view
triton_per_fused_leaky_relu_mean_view_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_mean_view_98', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (65536*x1)), rmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = 64.0
    tmp11 = tmp9 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckv62ou5ldlhemwyoghzfbcddsl3dpy4xbubixl2wxstecqzcdyq.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_99', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_136, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_137, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_138, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_139, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_140, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_141, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_142, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_143, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_144, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_145, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_146, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_147, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_148, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_149, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_150, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_151, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_153, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_154, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_155, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_156, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_158, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_159, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_160, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_161, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_162, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_164, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_166, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_167, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_168, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_170, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_171, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_172, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_174, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_175, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_176, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_177, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_178, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_179, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_180, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_181, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_182, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_183, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_184, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_185, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_186, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_187, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_188, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_189, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_191, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_192, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_194, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_195, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_196, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_198, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_199, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_200, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_201, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_202, (1000, 1024), (1024, 1))
    assert_size_stride(primals_203, (1000, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (64, ), (1, ))
    assert_size_stride(primals_218, (64, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (64, ), (1, ))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (128, ), (1, ))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (512, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (256, ), (1, ))
    assert_size_stride(primals_317, (256, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (256, ), (1, ))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (256, ), (1, ))
    assert_size_stride(primals_323, (256, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (256, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (256, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (256, ), (1, ))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (256, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (256, ), (1, ))
    assert_size_stride(primals_341, (256, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (256, ), (1, ))
    assert_size_stride(primals_347, (256, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (256, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (256, ), (1, ))
    assert_size_stride(primals_353, (256, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (256, ), (1, ))
    assert_size_stride(primals_356, (256, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (256, ), (1, ))
    assert_size_stride(primals_359, (256, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (256, ), (1, ))
    assert_size_stride(primals_362, (256, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (256, ), (1, ))
    assert_size_stride(primals_365, (256, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (512, ), (1, ))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_374, (1024, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (512, ), (1, ))
    assert_size_stride(primals_377, (512, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (512, ), (1, ))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (512, ), (1, ))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_386, (512, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (512, ), (1, ))
    assert_size_stride(primals_393, (), ())
    assert_size_stride(primals_394, (512, ), (1, ))
    assert_size_stride(primals_395, (512, ), (1, ))
    assert_size_stride(primals_396, (), ())
    assert_size_stride(primals_397, (512, ), (1, ))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_399, (), ())
    assert_size_stride(primals_400, (512, ), (1, ))
    assert_size_stride(primals_401, (512, ), (1, ))
    assert_size_stride(primals_402, (), ())
    assert_size_stride(primals_403, (1024, ), (1, ))
    assert_size_stride(primals_404, (1024, ), (1, ))
    assert_size_stride(primals_405, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_135, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_135
        buf1 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_136, buf1, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_136
        buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_139, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_139
        buf3 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_142, buf3, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_142
        buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_145, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_145
        buf5 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_147, buf5, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_147
        buf6 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_150, buf6, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_150
        buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_153, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_153
        buf8 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_155, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_155
        buf9 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_157, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_157
        buf10 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_159, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_159
        buf11 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_161, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_161
        buf12 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_163, buf12, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_163
        buf13 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_165, buf13, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_165
        buf14 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_167, buf14, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_167
        buf15 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_170, buf15, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_170
        buf16 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_173, buf16, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_173
        buf17 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_175, buf17, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_175
        buf18 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_177, buf18, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_177
        buf19 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_179, buf19, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_179
        buf20 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_181, buf20, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_181
        buf21 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_183, buf21, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_183
        buf22 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_185, buf22, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_185
        buf23 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_187, buf23, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_187
        buf24 = empty_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_190, buf24, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_190
        buf25 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_193, buf25, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_193
        buf26 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_195, buf26, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_195
        buf27 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_197, buf27, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_197
        buf28 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_199, buf28, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_199
        buf29 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_405, buf29, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del primals_405
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 256, 256), (2097152, 65536, 256, 1))
        buf31 = empty_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf30, buf31, 256, 65536, grid=grid(256, 65536), stream=stream0)
        buf32 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_12.run(buf31, buf32, buf33, buf34, 32768, 512, grid=grid(32768), stream=stream0)
        buf35 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf32, buf33, buf34, buf35, buf36, buf37, 256, 128, grid=grid(256), stream=stream0)
        buf38 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf41 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_14.run(buf35, buf36, buf37, primals_205, primals_206, buf38, buf39, buf41, primals_205, primals_206, 32, 8, grid=grid(32), stream=stream0)
        del primals_205
        del primals_206
        buf42 = reinterpret_tensor(buf30, (8, 32, 256, 256), (2097152, 1, 8192, 32), 0); del buf30  # reuse
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_15.run(buf43, buf31, buf38, buf39, primals_1, primals_2, 16777216, grid=grid(16777216), stream=stream0)
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf45 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf44, buf45, 512, 16384, grid=grid(512, 16384), stream=stream0)
        buf46 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf45, buf46, buf47, buf48, 65536, 128, grid=grid(65536), stream=stream0)
        buf49 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf46, buf47, buf48, buf49, buf50, buf51, 512, 128, grid=grid(512), stream=stream0)
        buf52 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf55 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf49, buf50, buf51, primals_208, primals_209, buf52, buf53, buf55, primals_208, primals_209, 64, 8, grid=grid(64), stream=stream0)
        del primals_208
        del primals_209
        buf56 = reinterpret_tensor(buf44, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf44  # reuse
        buf57 = buf56; del buf56  # reuse
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_20.run(buf57, buf45, buf52, buf53, primals_3, primals_4, 8388608, grid=grid(8388608), stream=stream0)
        del primals_4
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        buf59 = empty_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf58, buf59, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        buf60 = empty_strided((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_22.run(buf59, buf60, buf61, buf62, 131072, 128, grid=grid(131072), stream=stream0)
        buf63 = empty_strided((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), device='cuda', dtype=torch.float32)
        buf64 = empty_strided((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), device='cuda', dtype=torch.float32)
        buf65 = empty_strided((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf60, buf61, buf62, buf63, buf64, buf65, 1024, 128, grid=grid(1024), stream=stream0)
        del buf60
        del buf61
        del buf62
        buf66 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf69 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_24.run(buf63, buf64, buf65, primals_211, primals_212, buf66, buf67, buf69, primals_211, primals_212, 128, 8, grid=grid(128), stream=stream0)
        del primals_211
        del primals_212
        buf70 = reinterpret_tensor(buf58, (8, 128, 128, 128), (2097152, 1, 16384, 128), 0); del buf58  # reuse
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_25.run(buf59, buf66, buf67, primals_5, primals_6, buf70, 16777216, grid=grid(16777216), stream=stream0)
        del primals_6
        buf71 = empty((8, 128, 128, 128), device='cuda', dtype=torch.float32)
        buf852 = empty_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_18], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused_leaky_relu_leaky_relu_backward_26.run(buf70, buf71, buf852, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        buf72 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf71, buf72, 512, 16384, grid=grid(512, 16384), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf74 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf73, buf74, 256, 16384, grid=grid(256, 16384), stream=stream0)
        buf75 = buf34; del buf34  # reuse
        buf76 = buf33; del buf33  # reuse
        buf77 = buf32; del buf32  # reuse
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf74, buf75, buf76, buf77, 32768, 128, grid=grid(32768), stream=stream0)
        buf78 = buf37; del buf37  # reuse
        buf79 = buf36; del buf36  # reuse
        buf80 = buf35; del buf35  # reuse
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf75, buf76, buf77, buf78, buf79, buf80, 256, 128, grid=grid(256), stream=stream0)
        buf81 = buf39; del buf39  # reuse
        buf82 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf84 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf78, buf79, buf80, primals_214, primals_215, buf81, buf82, buf84, primals_214, primals_215, 32, 8, grid=grid(32), stream=stream0)
        del primals_214
        del primals_215
        buf85 = reinterpret_tensor(buf73, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf73  # reuse
        buf86 = buf85; del buf85  # reuse
        # Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_31.run(buf86, buf74, buf81, buf82, primals_7, primals_8, 4194304, grid=grid(4194304), stream=stream0)
        del buf82
        del primals_8
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf88 = buf72; del buf72  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf87, buf88, 512, 16384, grid=grid(512, 16384), stream=stream0)
        buf89 = buf48; del buf48  # reuse
        buf90 = buf47; del buf47  # reuse
        buf91 = buf46; del buf46  # reuse
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf88, buf89, buf90, buf91, 65536, 128, grid=grid(65536), stream=stream0)
        buf92 = buf51; del buf51  # reuse
        buf93 = buf50; del buf50  # reuse
        buf94 = buf49; del buf49  # reuse
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf89, buf90, buf91, buf92, buf93, buf94, 512, 128, grid=grid(512), stream=stream0)
        buf95 = buf53; del buf53  # reuse
        buf96 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf98 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf92, buf93, buf94, primals_217, primals_218, buf95, buf96, buf98, primals_217, primals_218, 64, 8, grid=grid(64), stream=stream0)
        del primals_217
        del primals_218
        buf100 = reinterpret_tensor(buf87, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf87  # reuse
        buf851 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_27, x_31, xb_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_32.run(buf88, buf95, buf96, primals_9, primals_10, buf71, buf100, buf851, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del primals_10
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf102 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf101, buf102, 512, 16384, grid=grid(512, 16384), stream=stream0)
        buf103 = buf91; del buf91  # reuse
        buf104 = buf90; del buf90  # reuse
        buf105 = buf89; del buf89  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf102, buf103, buf104, buf105, 65536, 128, grid=grid(65536), stream=stream0)
        buf106 = buf94; del buf94  # reuse
        buf107 = buf93; del buf93  # reuse
        buf108 = buf92; del buf92  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf103, buf104, buf105, buf106, buf107, buf108, 512, 128, grid=grid(512), stream=stream0)
        buf109 = buf96; del buf96  # reuse
        buf110 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf112 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf106, buf107, buf108, primals_220, primals_221, buf109, buf110, buf112, primals_220, primals_221, 64, 8, grid=grid(64), stream=stream0)
        del primals_220
        del primals_221
        buf113 = reinterpret_tensor(buf101, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf101  # reuse
        buf850 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_33.run(buf102, buf109, buf110, primals_11, primals_12, buf113, buf850, 8388608, grid=grid(8388608), stream=stream0)
        del primals_12
        buf114 = buf70; del buf70  # reuse
        # Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf71, buf113, buf114, 131072, 128, grid=grid(131072, 128), stream=stream0)
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf116 = buf113; del buf113  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf115, buf116, 512, 16384, grid=grid(512, 16384), stream=stream0)
        buf117 = buf105; del buf105  # reuse
        buf118 = buf104; del buf104  # reuse
        buf119 = buf103; del buf103  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf116, buf117, buf118, buf119, 65536, 128, grid=grid(65536), stream=stream0)
        buf120 = buf108; del buf108  # reuse
        buf121 = buf107; del buf107  # reuse
        buf122 = buf106; del buf106  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf117, buf118, buf119, buf120, buf121, buf122, 512, 128, grid=grid(512), stream=stream0)
        del buf117
        del buf118
        del buf119
        buf123 = buf110; del buf110  # reuse
        buf124 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf126 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf120, buf121, buf122, primals_223, primals_224, buf123, buf124, buf126, primals_223, primals_224, 64, 8, grid=grid(64), stream=stream0)
        del primals_223
        del primals_224
        buf127 = reinterpret_tensor(buf115, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf115  # reuse
        buf128 = buf127; del buf127  # reuse
        # Source Nodes: [out, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_20.run(buf128, buf116, buf123, buf124, primals_13, primals_14, 8388608, grid=grid(8388608), stream=stream0)
        del primals_14
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf130 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(buf129, buf130, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf131 = reinterpret_tensor(buf77, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf77  # reuse
        buf132 = reinterpret_tensor(buf76, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf76  # reuse
        buf133 = reinterpret_tensor(buf75, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf75  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf130, buf131, buf132, buf133, 32768, 128, grid=grid(32768), stream=stream0)
        buf134 = reinterpret_tensor(buf80, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf80  # reuse
        buf135 = reinterpret_tensor(buf79, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf79  # reuse
        buf136 = reinterpret_tensor(buf78, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf78  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf131, buf132, buf133, buf134, buf135, buf136, 256, 128, grid=grid(256), stream=stream0)
        buf137 = buf67; del buf67  # reuse
        buf138 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf140 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf134, buf135, buf136, primals_226, primals_227, buf137, buf138, buf140, primals_226, primals_227, 128, 2, grid=grid(128), stream=stream0)
        del primals_226
        del primals_227
        buf141 = reinterpret_tensor(buf129, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf129  # reuse
        buf142 = buf141; del buf141  # reuse
        # Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_39.run(buf142, buf130, buf137, buf138, primals_15, primals_16, 4194304, grid=grid(4194304), stream=stream0)
        del primals_16
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf144 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(buf143, buf144, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf145 = buf133; del buf133  # reuse
        buf146 = buf132; del buf132  # reuse
        buf147 = buf131; del buf131  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf144, buf145, buf146, buf147, 32768, 128, grid=grid(32768), stream=stream0)
        buf148 = buf136; del buf136  # reuse
        buf149 = buf135; del buf135  # reuse
        buf150 = buf134; del buf134  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf145, buf146, buf147, buf148, buf149, buf150, 256, 128, grid=grid(256), stream=stream0)
        buf151 = buf138; del buf138  # reuse
        buf152 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf154 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf148, buf149, buf150, primals_229, primals_230, buf151, buf152, buf154, primals_229, primals_230, 128, 2, grid=grid(128), stream=stream0)
        del primals_229
        del primals_230
        buf155 = reinterpret_tensor(buf143, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf143  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_40.run(buf144, buf151, buf152, primals_17, primals_18, buf155, 4194304, grid=grid(4194304), stream=stream0)
        del primals_18
        buf156 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        buf849 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_55], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused_leaky_relu_leaky_relu_backward_41.run(buf155, buf156, buf849, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf157 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf156, buf157, 512, 4096, grid=grid(512, 4096), stream=stream0)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf159 = buf157; del buf157  # reuse
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf158, buf159, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf160 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf161 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf162 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf159, buf160, buf161, buf162, 16384, 128, grid=grid(16384), stream=stream0)
        buf163 = reinterpret_tensor(buf152, (1, 64, 1, 1, 2), (128, 1, 128, 128, 64), 0); del buf152  # reuse
        buf164 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf160, buf161, buf162, buf163, buf164, buf165, 128, 128, grid=grid(128), stream=stream0)
        buf166 = buf124; del buf124  # reuse
        buf167 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf169 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf163, buf164, buf165, primals_232, primals_233, buf166, buf167, buf169, primals_232, primals_233, 64, 2, grid=grid(64), stream=stream0)
        del primals_232
        del primals_233
        buf170 = reinterpret_tensor(buf158, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf158  # reuse
        buf171 = buf170; del buf170  # reuse
        # Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_47.run(buf171, buf159, buf166, buf167, primals_19, primals_20, 2097152, grid=grid(2097152), stream=stream0)
        del primals_20
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf173 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf172, buf173, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf174 = buf162; del buf162  # reuse
        buf175 = buf161; del buf161  # reuse
        buf176 = buf160; del buf160  # reuse
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf173, buf174, buf175, buf176, 16384, 128, grid=grid(16384), stream=stream0)
        buf177 = buf165; del buf165  # reuse
        buf178 = buf164; del buf164  # reuse
        buf179 = buf163; del buf163  # reuse
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf174, buf175, buf176, buf177, buf178, buf179, 128, 128, grid=grid(128), stream=stream0)
        buf180 = buf167; del buf167  # reuse
        buf181 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf183 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf177, buf178, buf179, primals_235, primals_236, buf180, buf181, buf183, primals_235, primals_236, 64, 2, grid=grid(64), stream=stream0)
        del primals_235
        del primals_236
        buf185 = reinterpret_tensor(buf172, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf172  # reuse
        buf848 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_2, x_64, x_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_48.run(buf173, buf180, buf181, primals_21, primals_22, buf156, buf185, buf848, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del primals_22
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf187 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf186, buf187, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf188 = buf176; del buf176  # reuse
        buf189 = buf175; del buf175  # reuse
        buf190 = buf174; del buf174  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf187, buf188, buf189, buf190, 16384, 128, grid=grid(16384), stream=stream0)
        buf191 = buf179; del buf179  # reuse
        buf192 = buf178; del buf178  # reuse
        buf193 = buf177; del buf177  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf188, buf189, buf190, buf191, buf192, buf193, 128, 128, grid=grid(128), stream=stream0)
        buf194 = buf181; del buf181  # reuse
        buf195 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf197 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf191, buf192, buf193, primals_238, primals_239, buf194, buf195, buf197, primals_238, primals_239, 64, 2, grid=grid(64), stream=stream0)
        del primals_238
        del primals_239
        buf198 = reinterpret_tensor(buf186, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf186  # reuse
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [x_71, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_47.run(buf199, buf187, buf194, buf195, primals_23, primals_24, 2097152, grid=grid(2097152), stream=stream0)
        del primals_24
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf201 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf200, buf201, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf202 = buf190; del buf190  # reuse
        buf203 = buf189; del buf189  # reuse
        buf204 = buf188; del buf188  # reuse
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf201, buf202, buf203, buf204, 16384, 128, grid=grid(16384), stream=stream0)
        buf205 = buf193; del buf193  # reuse
        buf206 = buf192; del buf192  # reuse
        buf207 = buf191; del buf191  # reuse
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf202, buf203, buf204, buf205, buf206, buf207, 128, 128, grid=grid(128), stream=stream0)
        buf208 = buf195; del buf195  # reuse
        buf209 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf211 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf205, buf206, buf207, primals_241, primals_242, buf208, buf209, buf211, primals_241, primals_242, 64, 2, grid=grid(64), stream=stream0)
        del primals_241
        del primals_242
        buf213 = reinterpret_tensor(buf200, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf200  # reuse
        buf847 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_78, x_82, xb_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_49.run(buf201, buf208, buf209, primals_25, primals_26, buf185, buf213, buf847, 2097152, grid=grid(2097152), stream=stream0)
        del primals_26
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf215 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf214, buf215, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf216 = buf204; del buf204  # reuse
        buf217 = buf203; del buf203  # reuse
        buf218 = buf202; del buf202  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf215, buf216, buf217, buf218, 16384, 128, grid=grid(16384), stream=stream0)
        buf219 = buf207; del buf207  # reuse
        buf220 = buf206; del buf206  # reuse
        buf221 = buf205; del buf205  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf216, buf217, buf218, buf219, buf220, buf221, 128, 128, grid=grid(128), stream=stream0)
        buf222 = buf209; del buf209  # reuse
        buf223 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf225 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf219, buf220, buf221, primals_244, primals_245, buf222, buf223, buf225, primals_244, primals_245, 64, 2, grid=grid(64), stream=stream0)
        del primals_244
        del primals_245
        buf226 = reinterpret_tensor(buf214, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf214  # reuse
        buf846 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_85, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_50.run(buf215, buf222, buf223, primals_27, primals_28, buf226, buf846, 2097152, grid=grid(2097152), stream=stream0)
        del buf223
        del primals_28
        buf227 = buf155; del buf155  # reuse
        # Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf156, buf226, buf227, 32768, 128, grid=grid(32768, 128), stream=stream0)
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf229 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(buf228, buf229, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf230 = buf147; del buf147  # reuse
        buf231 = buf146; del buf146  # reuse
        buf232 = buf145; del buf145  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf229, buf230, buf231, buf232, 32768, 128, grid=grid(32768), stream=stream0)
        buf233 = buf150; del buf150  # reuse
        buf234 = buf149; del buf149  # reuse
        buf235 = buf148; del buf148  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf230, buf231, buf232, buf233, buf234, buf235, 256, 128, grid=grid(256), stream=stream0)
        del buf230
        del buf231
        del buf232
        buf236 = reinterpret_tensor(buf221, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf221  # reuse
        buf237 = reinterpret_tensor(buf220, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf220  # reuse
        buf239 = reinterpret_tensor(buf219, (128, ), (1, ), 0); del buf219  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf233, buf234, buf235, primals_247, primals_248, buf236, buf237, buf239, primals_247, primals_248, 128, 2, grid=grid(128), stream=stream0)
        del primals_247
        del primals_248
        buf240 = reinterpret_tensor(buf228, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf228  # reuse
        buf241 = buf240; del buf240  # reuse
        # Source Nodes: [out_1, x_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_39.run(buf241, buf229, buf236, buf237, primals_29, primals_30, 4194304, grid=grid(4194304), stream=stream0)
        del primals_30
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf243 = reinterpret_tensor(buf226, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf226  # reuse
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf242, buf243, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf244 = reinterpret_tensor(buf218, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf218  # reuse
        buf245 = reinterpret_tensor(buf217, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf217  # reuse
        buf246 = reinterpret_tensor(buf216, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf216  # reuse
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf243, buf244, buf245, buf246, 16384, 128, grid=grid(16384), stream=stream0)
        buf247 = reinterpret_tensor(buf235, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf235  # reuse
        buf248 = reinterpret_tensor(buf234, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf234  # reuse
        buf250 = reinterpret_tensor(buf233, (256, ), (1, ), 0); del buf233  # reuse
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf244, buf245, buf246, primals_250, primals_251, buf247, buf248, buf250, primals_250, primals_251, 256, 64, grid=grid(256), stream=stream0)
        del primals_250
        del primals_251
        buf251 = reinterpret_tensor(buf242, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf242  # reuse
        buf252 = buf251; del buf251  # reuse
        # Source Nodes: [x_95, x_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_55.run(buf252, buf243, buf247, buf248, primals_31, primals_32, 2097152, grid=grid(2097152), stream=stream0)
        del primals_32
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf254 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf253, buf254, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf255 = buf246; del buf246  # reuse
        buf256 = buf245; del buf245  # reuse
        buf257 = buf244; del buf244  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf254, buf255, buf256, buf257, 16384, 128, grid=grid(16384), stream=stream0)
        buf258 = buf248; del buf248  # reuse
        buf259 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf261 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf255, buf256, buf257, primals_253, primals_254, buf258, buf259, buf261, primals_253, primals_254, 256, 64, grid=grid(256), stream=stream0)
        del primals_253
        del primals_254
        buf262 = reinterpret_tensor(buf253, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf253  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_56.run(buf254, buf258, buf259, primals_33, primals_34, buf262, 2097152, grid=grid(2097152), stream=stream0)
        del primals_34
        buf263 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        buf845 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_106], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused_leaky_relu_leaky_relu_backward_57.run(buf262, buf263, buf845, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf264 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf263, buf264, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf266 = buf264; del buf264  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf265, buf266, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf267 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf268 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf269 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf266, buf267, buf268, buf269, 8192, 128, grid=grid(8192), stream=stream0)
        buf270 = buf237; del buf237  # reuse
        buf271 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf273 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf267, buf268, buf269, primals_256, primals_257, buf270, buf271, buf273, primals_256, primals_257, 128, 64, grid=grid(128), stream=stream0)
        del primals_256
        del primals_257
        buf274 = reinterpret_tensor(buf265, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf265  # reuse
        buf275 = buf274; del buf274  # reuse
        # Source Nodes: [x_108, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf275, buf266, buf270, buf271, primals_35, primals_36, 1048576, grid=grid(1048576), stream=stream0)
        del primals_36
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf277 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf276, buf277, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf278 = buf269; del buf269  # reuse
        buf279 = buf268; del buf268  # reuse
        buf280 = buf267; del buf267  # reuse
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf277, buf278, buf279, buf280, 8192, 128, grid=grid(8192), stream=stream0)
        buf281 = buf271; del buf271  # reuse
        buf282 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf284 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf278, buf279, buf280, primals_259, primals_260, buf281, buf282, buf284, primals_259, primals_260, 128, 64, grid=grid(128), stream=stream0)
        del primals_259
        del primals_260
        buf286 = reinterpret_tensor(buf276, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf276  # reuse
        buf844 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_4, x_115, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_63.run(buf277, buf281, buf282, primals_37, primals_38, buf263, buf286, buf844, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del primals_38
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf288 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf287, buf288, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf289 = buf280; del buf280  # reuse
        buf290 = buf279; del buf279  # reuse
        buf291 = buf278; del buf278  # reuse
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf288, buf289, buf290, buf291, 8192, 128, grid=grid(8192), stream=stream0)
        buf292 = buf282; del buf282  # reuse
        buf293 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf295 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf289, buf290, buf291, primals_262, primals_263, buf292, buf293, buf295, primals_262, primals_263, 128, 64, grid=grid(128), stream=stream0)
        del primals_262
        del primals_263
        buf296 = reinterpret_tensor(buf287, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf287  # reuse
        buf297 = buf296; del buf296  # reuse
        # Source Nodes: [x_122, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf297, buf288, buf292, buf293, primals_39, primals_40, 1048576, grid=grid(1048576), stream=stream0)
        del primals_40
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf299 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf298, buf299, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf300 = buf291; del buf291  # reuse
        buf301 = buf290; del buf290  # reuse
        buf302 = buf289; del buf289  # reuse
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf299, buf300, buf301, buf302, 8192, 128, grid=grid(8192), stream=stream0)
        buf303 = buf293; del buf293  # reuse
        buf304 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf306 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf300, buf301, buf302, primals_265, primals_266, buf303, buf304, buf306, primals_265, primals_266, 128, 64, grid=grid(128), stream=stream0)
        del primals_265
        del primals_266
        buf308 = reinterpret_tensor(buf298, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf298  # reuse
        buf843 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_5, x_129, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf299, buf303, buf304, primals_41, primals_42, buf286, buf308, buf843, 1048576, grid=grid(1048576), stream=stream0)
        del primals_42
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf310 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf309, buf310, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf311 = buf302; del buf302  # reuse
        buf312 = buf301; del buf301  # reuse
        buf313 = buf300; del buf300  # reuse
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf310, buf311, buf312, buf313, 8192, 128, grid=grid(8192), stream=stream0)
        buf314 = buf304; del buf304  # reuse
        buf315 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf317 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf311, buf312, buf313, primals_268, primals_269, buf314, buf315, buf317, primals_268, primals_269, 128, 64, grid=grid(128), stream=stream0)
        del primals_268
        del primals_269
        buf318 = reinterpret_tensor(buf309, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf309  # reuse
        buf319 = buf318; del buf318  # reuse
        # Source Nodes: [x_136, x_140], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf319, buf310, buf314, buf315, primals_43, primals_44, 1048576, grid=grid(1048576), stream=stream0)
        del primals_44
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf321 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf320, buf321, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf322 = buf313; del buf313  # reuse
        buf323 = buf312; del buf312  # reuse
        buf324 = buf311; del buf311  # reuse
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf321, buf322, buf323, buf324, 8192, 128, grid=grid(8192), stream=stream0)
        buf325 = buf315; del buf315  # reuse
        buf326 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf328 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf322, buf323, buf324, primals_271, primals_272, buf325, buf326, buf328, primals_271, primals_272, 128, 64, grid=grid(128), stream=stream0)
        del primals_271
        del primals_272
        buf330 = reinterpret_tensor(buf320, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf320  # reuse
        buf842 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_6, x_143, x_147], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf321, buf325, buf326, primals_45, primals_46, buf308, buf330, buf842, 1048576, grid=grid(1048576), stream=stream0)
        del primals_46
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf332 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf331, buf332, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf333 = buf324; del buf324  # reuse
        buf334 = buf323; del buf323  # reuse
        buf335 = buf322; del buf322  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf332, buf333, buf334, buf335, 8192, 128, grid=grid(8192), stream=stream0)
        buf336 = buf326; del buf326  # reuse
        buf337 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf339 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf333, buf334, buf335, primals_274, primals_275, buf336, buf337, buf339, primals_274, primals_275, 128, 64, grid=grid(128), stream=stream0)
        del primals_274
        del primals_275
        buf340 = reinterpret_tensor(buf331, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf331  # reuse
        buf341 = buf340; del buf340  # reuse
        # Source Nodes: [x_150, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf341, buf332, buf336, buf337, primals_47, primals_48, 1048576, grid=grid(1048576), stream=stream0)
        del primals_48
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf343 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf342, buf343, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf344 = buf335; del buf335  # reuse
        buf345 = buf334; del buf334  # reuse
        buf346 = buf333; del buf333  # reuse
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf343, buf344, buf345, buf346, 8192, 128, grid=grid(8192), stream=stream0)
        buf347 = buf337; del buf337  # reuse
        buf348 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf350 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf344, buf345, buf346, primals_277, primals_278, buf347, buf348, buf350, primals_277, primals_278, 128, 64, grid=grid(128), stream=stream0)
        del primals_277
        del primals_278
        buf352 = reinterpret_tensor(buf342, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf342  # reuse
        buf841 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_7, x_157, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf343, buf347, buf348, primals_49, primals_50, buf330, buf352, buf841, 1048576, grid=grid(1048576), stream=stream0)
        del primals_50
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf354 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf353, buf354, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf355 = buf346; del buf346  # reuse
        buf356 = buf345; del buf345  # reuse
        buf357 = buf344; del buf344  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf354, buf355, buf356, buf357, 8192, 128, grid=grid(8192), stream=stream0)
        buf358 = buf348; del buf348  # reuse
        buf359 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf361 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf355, buf356, buf357, primals_280, primals_281, buf358, buf359, buf361, primals_280, primals_281, 128, 64, grid=grid(128), stream=stream0)
        del primals_280
        del primals_281
        buf362 = reinterpret_tensor(buf353, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf353  # reuse
        buf363 = buf362; del buf362  # reuse
        # Source Nodes: [x_164, x_168], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf363, buf354, buf358, buf359, primals_51, primals_52, 1048576, grid=grid(1048576), stream=stream0)
        del primals_52
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf365 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf364, buf365, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf366 = buf357; del buf357  # reuse
        buf367 = buf356; del buf356  # reuse
        buf368 = buf355; del buf355  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf365, buf366, buf367, buf368, 8192, 128, grid=grid(8192), stream=stream0)
        buf369 = buf359; del buf359  # reuse
        buf370 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf372 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf366, buf367, buf368, primals_283, primals_284, buf369, buf370, buf372, primals_283, primals_284, 128, 64, grid=grid(128), stream=stream0)
        del primals_283
        del primals_284
        buf374 = reinterpret_tensor(buf364, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf364  # reuse
        buf840 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_8, x_171, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf365, buf369, buf370, primals_53, primals_54, buf352, buf374, buf840, 1048576, grid=grid(1048576), stream=stream0)
        del primals_54
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf376 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf375, buf376, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf377 = buf368; del buf368  # reuse
        buf378 = buf367; del buf367  # reuse
        buf379 = buf366; del buf366  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf376, buf377, buf378, buf379, 8192, 128, grid=grid(8192), stream=stream0)
        buf380 = buf370; del buf370  # reuse
        buf381 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf383 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf377, buf378, buf379, primals_286, primals_287, buf380, buf381, buf383, primals_286, primals_287, 128, 64, grid=grid(128), stream=stream0)
        del primals_286
        del primals_287
        buf384 = reinterpret_tensor(buf375, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf375  # reuse
        buf385 = buf384; del buf384  # reuse
        # Source Nodes: [x_178, x_182], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf385, buf376, buf380, buf381, primals_55, primals_56, 1048576, grid=grid(1048576), stream=stream0)
        del primals_56
        # Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf387 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf386, buf387, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf388 = buf379; del buf379  # reuse
        buf389 = buf378; del buf378  # reuse
        buf390 = buf377; del buf377  # reuse
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf387, buf388, buf389, buf390, 8192, 128, grid=grid(8192), stream=stream0)
        buf391 = buf381; del buf381  # reuse
        buf392 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf394 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf388, buf389, buf390, primals_289, primals_290, buf391, buf392, buf394, primals_289, primals_290, 128, 64, grid=grid(128), stream=stream0)
        del primals_289
        del primals_290
        buf396 = reinterpret_tensor(buf386, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf386  # reuse
        buf839 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_9, x_185, x_189], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf387, buf391, buf392, primals_57, primals_58, buf374, buf396, buf839, 1048576, grid=grid(1048576), stream=stream0)
        del primals_58
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf398 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf397, buf398, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf399 = buf390; del buf390  # reuse
        buf400 = buf389; del buf389  # reuse
        buf401 = buf388; del buf388  # reuse
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf398, buf399, buf400, buf401, 8192, 128, grid=grid(8192), stream=stream0)
        buf402 = buf392; del buf392  # reuse
        buf403 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf405 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf399, buf400, buf401, primals_292, primals_293, buf402, buf403, buf405, primals_292, primals_293, 128, 64, grid=grid(128), stream=stream0)
        del primals_292
        del primals_293
        buf406 = reinterpret_tensor(buf397, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf397  # reuse
        buf407 = buf406; del buf406  # reuse
        # Source Nodes: [x_192, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf407, buf398, buf402, buf403, primals_59, primals_60, 1048576, grid=grid(1048576), stream=stream0)
        del primals_60
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf409 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf408, buf409, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf410 = buf401; del buf401  # reuse
        buf411 = buf400; del buf400  # reuse
        buf412 = buf399; del buf399  # reuse
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf409, buf410, buf411, buf412, 8192, 128, grid=grid(8192), stream=stream0)
        buf413 = buf403; del buf403  # reuse
        buf414 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf416 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf410, buf411, buf412, primals_295, primals_296, buf413, buf414, buf416, primals_295, primals_296, 128, 64, grid=grid(128), stream=stream0)
        del primals_295
        del primals_296
        buf418 = reinterpret_tensor(buf408, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf408  # reuse
        buf838 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_10, x_199, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf409, buf413, buf414, primals_61, primals_62, buf396, buf418, buf838, 1048576, grid=grid(1048576), stream=stream0)
        del primals_62
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf420 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf419, buf420, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf421 = buf412; del buf412  # reuse
        buf422 = buf411; del buf411  # reuse
        buf423 = buf410; del buf410  # reuse
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf420, buf421, buf422, buf423, 8192, 128, grid=grid(8192), stream=stream0)
        buf424 = buf414; del buf414  # reuse
        buf425 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf427 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf421, buf422, buf423, primals_298, primals_299, buf424, buf425, buf427, primals_298, primals_299, 128, 64, grid=grid(128), stream=stream0)
        del primals_298
        del primals_299
        buf428 = reinterpret_tensor(buf419, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf419  # reuse
        buf429 = buf428; del buf428  # reuse
        # Source Nodes: [x_206, x_210], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_62.run(buf429, buf420, buf424, buf425, primals_63, primals_64, 1048576, grid=grid(1048576), stream=stream0)
        del primals_64
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf431 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf430, buf431, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf432 = buf423; del buf423  # reuse
        buf433 = buf422; del buf422  # reuse
        buf434 = buf421; del buf421  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf431, buf432, buf433, buf434, 8192, 128, grid=grid(8192), stream=stream0)
        buf435 = buf425; del buf425  # reuse
        buf436 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf438 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf432, buf433, buf434, primals_301, primals_302, buf435, buf436, buf438, primals_301, primals_302, 128, 64, grid=grid(128), stream=stream0)
        del primals_301
        del primals_302
        buf440 = reinterpret_tensor(buf430, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf430  # reuse
        buf837 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_213, x_217, xb_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_64.run(buf431, buf435, buf436, primals_65, primals_66, buf418, buf440, buf837, 1048576, grid=grid(1048576), stream=stream0)
        del primals_66
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf442 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf441, buf442, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf443 = buf434; del buf434  # reuse
        buf444 = buf433; del buf433  # reuse
        buf445 = buf432; del buf432  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf442, buf443, buf444, buf445, 8192, 128, grid=grid(8192), stream=stream0)
        buf446 = buf436; del buf436  # reuse
        buf447 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf449 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf443, buf444, buf445, primals_304, primals_305, buf446, buf447, buf449, primals_304, primals_305, 128, 64, grid=grid(128), stream=stream0)
        del primals_304
        del primals_305
        buf450 = reinterpret_tensor(buf441, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf441  # reuse
        buf836 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_220, x_223], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_65.run(buf442, buf446, buf447, primals_67, primals_68, buf450, buf836, 1048576, grid=grid(1048576), stream=stream0)
        del buf447
        del primals_68
        buf451 = buf262; del buf262  # reuse
        # Source Nodes: [cat_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_66.run(buf263, buf450, buf451, 8192, 256, grid=grid(8192, 256), stream=stream0)
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf453 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf452, buf453, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf454 = buf257; del buf257  # reuse
        buf455 = buf256; del buf256  # reuse
        buf456 = buf255; del buf255  # reuse
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf453, buf454, buf455, buf456, 16384, 128, grid=grid(16384), stream=stream0)
        buf457 = buf259; del buf259  # reuse
        buf458 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf460 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf454, buf455, buf456, primals_307, primals_308, buf457, buf458, buf460, primals_307, primals_308, 256, 64, grid=grid(256), stream=stream0)
        del buf454
        del buf455
        del buf456
        del primals_307
        del primals_308
        buf461 = reinterpret_tensor(buf452, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf452  # reuse
        buf462 = buf461; del buf461  # reuse
        # Source Nodes: [out_2, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_55.run(buf462, buf453, buf457, buf458, primals_69, primals_70, 2097152, grid=grid(2097152), stream=stream0)
        del primals_70
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, buf15, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf464 = reinterpret_tensor(buf450, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf450  # reuse
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf463, buf464, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf465 = reinterpret_tensor(buf445, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf445  # reuse
        buf466 = reinterpret_tensor(buf444, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf444  # reuse
        buf467 = reinterpret_tensor(buf443, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf443  # reuse
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf464, buf465, buf466, buf467, 8192, 128, grid=grid(8192), stream=stream0)
        buf468 = reinterpret_tensor(buf122, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf122  # reuse
        buf469 = reinterpret_tensor(buf121, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf121  # reuse
        buf471 = reinterpret_tensor(buf120, (512, ), (1, ), 0); del buf120  # reuse
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf465, buf466, buf467, primals_310, primals_311, buf468, buf469, buf471, primals_310, primals_311, 512, 16, grid=grid(512), stream=stream0)
        del primals_310
        del primals_311
        buf472 = reinterpret_tensor(buf463, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf463  # reuse
        buf473 = buf472; del buf472  # reuse
        # Source Nodes: [x_230, x_233], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_70.run(buf473, buf464, buf468, buf469, primals_71, primals_72, 1048576, grid=grid(1048576), stream=stream0)
        del primals_72
        # Source Nodes: [x_236], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf475 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf474, buf475, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf476 = buf467; del buf467  # reuse
        buf477 = buf466; del buf466  # reuse
        buf478 = buf465; del buf465  # reuse
        # Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf475, buf476, buf477, buf478, 8192, 128, grid=grid(8192), stream=stream0)
        buf479 = buf469; del buf469  # reuse
        buf480 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf482 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf476, buf477, buf478, primals_313, primals_314, buf479, buf480, buf482, primals_313, primals_314, 512, 16, grid=grid(512), stream=stream0)
        del primals_313
        del primals_314
        buf483 = reinterpret_tensor(buf474, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf474  # reuse
        # Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_71.run(buf475, buf479, buf480, primals_73, primals_74, buf483, 1048576, grid=grid(1048576), stream=stream0)
        del primals_74
        buf484 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        buf835 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_241], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused_leaky_relu_leaky_relu_backward_72.run(buf483, buf484, buf835, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf485 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf484, buf485, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf487 = buf485; del buf485  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf486, buf487, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf488 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf489 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf490 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf487, buf488, buf489, buf490, 4096, 128, grid=grid(4096), stream=stream0)
        buf491 = buf458; del buf458  # reuse
        buf492 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf494 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf488, buf489, buf490, primals_316, primals_317, buf491, buf492, buf494, primals_316, primals_317, 256, 16, grid=grid(256), stream=stream0)
        del primals_316
        del primals_317
        buf495 = reinterpret_tensor(buf486, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf486  # reuse
        buf496 = buf495; del buf495  # reuse
        # Source Nodes: [x_243, x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf496, buf487, buf491, buf492, primals_75, primals_76, 524288, grid=grid(524288), stream=stream0)
        del primals_76
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf498 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf497, buf498, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf499 = buf490; del buf490  # reuse
        buf500 = buf489; del buf489  # reuse
        buf501 = buf488; del buf488  # reuse
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf498, buf499, buf500, buf501, 4096, 128, grid=grid(4096), stream=stream0)
        buf502 = buf492; del buf492  # reuse
        buf503 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf505 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf499, buf500, buf501, primals_319, primals_320, buf502, buf503, buf505, primals_319, primals_320, 256, 16, grid=grid(256), stream=stream0)
        del primals_319
        del primals_320
        buf507 = reinterpret_tensor(buf497, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf497  # reuse
        buf834 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_12, x_250, x_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_78.run(buf498, buf502, buf503, primals_77, primals_78, buf484, buf507, buf834, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del primals_78
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf509 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf508, buf509, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf510 = buf501; del buf501  # reuse
        buf511 = buf500; del buf500  # reuse
        buf512 = buf499; del buf499  # reuse
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf509, buf510, buf511, buf512, 4096, 128, grid=grid(4096), stream=stream0)
        buf513 = buf503; del buf503  # reuse
        buf514 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf516 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf510, buf511, buf512, primals_322, primals_323, buf513, buf514, buf516, primals_322, primals_323, 256, 16, grid=grid(256), stream=stream0)
        del primals_322
        del primals_323
        buf517 = reinterpret_tensor(buf508, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf508  # reuse
        buf518 = buf517; del buf517  # reuse
        # Source Nodes: [x_257, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf518, buf509, buf513, buf514, primals_79, primals_80, 524288, grid=grid(524288), stream=stream0)
        del primals_80
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf520 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf519, buf520, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf521 = buf512; del buf512  # reuse
        buf522 = buf511; del buf511  # reuse
        buf523 = buf510; del buf510  # reuse
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf520, buf521, buf522, buf523, 4096, 128, grid=grid(4096), stream=stream0)
        buf524 = buf514; del buf514  # reuse
        buf525 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf527 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf521, buf522, buf523, primals_325, primals_326, buf524, buf525, buf527, primals_325, primals_326, 256, 16, grid=grid(256), stream=stream0)
        del primals_325
        del primals_326
        buf529 = reinterpret_tensor(buf519, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf519  # reuse
        buf833 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_13, x_264, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf520, buf524, buf525, primals_81, primals_82, buf507, buf529, buf833, 524288, grid=grid(524288), stream=stream0)
        del primals_82
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf531 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf530, buf531, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf532 = buf523; del buf523  # reuse
        buf533 = buf522; del buf522  # reuse
        buf534 = buf521; del buf521  # reuse
        # Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf531, buf532, buf533, buf534, 4096, 128, grid=grid(4096), stream=stream0)
        buf535 = buf525; del buf525  # reuse
        buf536 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf538 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf532, buf533, buf534, primals_328, primals_329, buf535, buf536, buf538, primals_328, primals_329, 256, 16, grid=grid(256), stream=stream0)
        del primals_328
        del primals_329
        buf539 = reinterpret_tensor(buf530, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf530  # reuse
        buf540 = buf539; del buf539  # reuse
        # Source Nodes: [x_271, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf540, buf531, buf535, buf536, primals_83, primals_84, 524288, grid=grid(524288), stream=stream0)
        del primals_84
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf542 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf541, buf542, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf543 = buf534; del buf534  # reuse
        buf544 = buf533; del buf533  # reuse
        buf545 = buf532; del buf532  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf542, buf543, buf544, buf545, 4096, 128, grid=grid(4096), stream=stream0)
        buf546 = buf536; del buf536  # reuse
        buf547 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf549 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf543, buf544, buf545, primals_331, primals_332, buf546, buf547, buf549, primals_331, primals_332, 256, 16, grid=grid(256), stream=stream0)
        del primals_331
        del primals_332
        buf551 = reinterpret_tensor(buf541, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf541  # reuse
        buf832 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_14, x_278, x_282], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf542, buf546, buf547, primals_85, primals_86, buf529, buf551, buf832, 524288, grid=grid(524288), stream=stream0)
        del primals_86
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf553 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf552, buf553, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf554 = buf545; del buf545  # reuse
        buf555 = buf544; del buf544  # reuse
        buf556 = buf543; del buf543  # reuse
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf553, buf554, buf555, buf556, 4096, 128, grid=grid(4096), stream=stream0)
        buf557 = buf547; del buf547  # reuse
        buf558 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf560 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf554, buf555, buf556, primals_334, primals_335, buf557, buf558, buf560, primals_334, primals_335, 256, 16, grid=grid(256), stream=stream0)
        del primals_334
        del primals_335
        buf561 = reinterpret_tensor(buf552, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf552  # reuse
        buf562 = buf561; del buf561  # reuse
        # Source Nodes: [x_285, x_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf562, buf553, buf557, buf558, primals_87, primals_88, 524288, grid=grid(524288), stream=stream0)
        del primals_88
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf564 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf563, buf564, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf565 = buf556; del buf556  # reuse
        buf566 = buf555; del buf555  # reuse
        buf567 = buf554; del buf554  # reuse
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf564, buf565, buf566, buf567, 4096, 128, grid=grid(4096), stream=stream0)
        buf568 = buf558; del buf558  # reuse
        buf569 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf571 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf565, buf566, buf567, primals_337, primals_338, buf568, buf569, buf571, primals_337, primals_338, 256, 16, grid=grid(256), stream=stream0)
        del primals_337
        del primals_338
        buf573 = reinterpret_tensor(buf563, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf563  # reuse
        buf831 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_15, x_292, x_296], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf564, buf568, buf569, primals_89, primals_90, buf551, buf573, buf831, 524288, grid=grid(524288), stream=stream0)
        del primals_90
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf575 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf574, buf575, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf576 = buf567; del buf567  # reuse
        buf577 = buf566; del buf566  # reuse
        buf578 = buf565; del buf565  # reuse
        # Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf575, buf576, buf577, buf578, 4096, 128, grid=grid(4096), stream=stream0)
        buf579 = buf569; del buf569  # reuse
        buf580 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf582 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf576, buf577, buf578, primals_340, primals_341, buf579, buf580, buf582, primals_340, primals_341, 256, 16, grid=grid(256), stream=stream0)
        del primals_340
        del primals_341
        buf583 = reinterpret_tensor(buf574, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf574  # reuse
        buf584 = buf583; del buf583  # reuse
        # Source Nodes: [x_299, x_303], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf584, buf575, buf579, buf580, primals_91, primals_92, 524288, grid=grid(524288), stream=stream0)
        del primals_92
        # Source Nodes: [x_305], Original ATen: [aten.convolution]
        buf585 = extern_kernels.convolution(buf584, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf585, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf586 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf585, buf586, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf587 = buf578; del buf578  # reuse
        buf588 = buf577; del buf577  # reuse
        buf589 = buf576; del buf576  # reuse
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf586, buf587, buf588, buf589, 4096, 128, grid=grid(4096), stream=stream0)
        buf590 = buf580; del buf580  # reuse
        buf591 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf593 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf587, buf588, buf589, primals_343, primals_344, buf590, buf591, buf593, primals_343, primals_344, 256, 16, grid=grid(256), stream=stream0)
        del primals_343
        del primals_344
        buf595 = reinterpret_tensor(buf585, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf585  # reuse
        buf830 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_16, x_306, x_310], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf586, buf590, buf591, primals_93, primals_94, buf573, buf595, buf830, 524288, grid=grid(524288), stream=stream0)
        del primals_94
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf597 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf596, buf597, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf598 = buf589; del buf589  # reuse
        buf599 = buf588; del buf588  # reuse
        buf600 = buf587; del buf587  # reuse
        # Source Nodes: [x_313], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf597, buf598, buf599, buf600, 4096, 128, grid=grid(4096), stream=stream0)
        buf601 = buf591; del buf591  # reuse
        buf602 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf604 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_313], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf598, buf599, buf600, primals_346, primals_347, buf601, buf602, buf604, primals_346, primals_347, 256, 16, grid=grid(256), stream=stream0)
        del primals_346
        del primals_347
        buf605 = reinterpret_tensor(buf596, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf596  # reuse
        buf606 = buf605; del buf605  # reuse
        # Source Nodes: [x_313, x_317], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf606, buf597, buf601, buf602, primals_95, primals_96, 524288, grid=grid(524288), stream=stream0)
        del primals_96
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf608 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf607, buf608, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf609 = buf600; del buf600  # reuse
        buf610 = buf599; del buf599  # reuse
        buf611 = buf598; del buf598  # reuse
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf608, buf609, buf610, buf611, 4096, 128, grid=grid(4096), stream=stream0)
        buf612 = buf602; del buf602  # reuse
        buf613 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf615 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf609, buf610, buf611, primals_349, primals_350, buf612, buf613, buf615, primals_349, primals_350, 256, 16, grid=grid(256), stream=stream0)
        del primals_349
        del primals_350
        buf617 = reinterpret_tensor(buf607, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf607  # reuse
        buf829 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_17, x_320, x_324], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf608, buf612, buf613, primals_97, primals_98, buf595, buf617, buf829, 524288, grid=grid(524288), stream=stream0)
        del primals_98
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf619 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf618, buf619, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf620 = buf611; del buf611  # reuse
        buf621 = buf610; del buf610  # reuse
        buf622 = buf609; del buf609  # reuse
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf619, buf620, buf621, buf622, 4096, 128, grid=grid(4096), stream=stream0)
        buf623 = buf613; del buf613  # reuse
        buf624 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf626 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf620, buf621, buf622, primals_352, primals_353, buf623, buf624, buf626, primals_352, primals_353, 256, 16, grid=grid(256), stream=stream0)
        del primals_352
        del primals_353
        buf627 = reinterpret_tensor(buf618, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf618  # reuse
        buf628 = buf627; del buf627  # reuse
        # Source Nodes: [x_327, x_331], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf628, buf619, buf623, buf624, primals_99, primals_100, 524288, grid=grid(524288), stream=stream0)
        del primals_100
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf629 = extern_kernels.convolution(buf628, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf629, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf630 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf629, buf630, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf631 = buf622; del buf622  # reuse
        buf632 = buf621; del buf621  # reuse
        buf633 = buf620; del buf620  # reuse
        # Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf630, buf631, buf632, buf633, 4096, 128, grid=grid(4096), stream=stream0)
        buf634 = buf624; del buf624  # reuse
        buf635 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf637 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf631, buf632, buf633, primals_355, primals_356, buf634, buf635, buf637, primals_355, primals_356, 256, 16, grid=grid(256), stream=stream0)
        del primals_355
        del primals_356
        buf639 = reinterpret_tensor(buf629, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf629  # reuse
        buf828 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_18, x_334, x_338], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf630, buf634, buf635, primals_101, primals_102, buf617, buf639, buf828, 524288, grid=grid(524288), stream=stream0)
        del primals_102
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        buf640 = extern_kernels.convolution(buf639, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf640, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf641 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf640, buf641, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf642 = buf633; del buf633  # reuse
        buf643 = buf632; del buf632  # reuse
        buf644 = buf631; del buf631  # reuse
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf641, buf642, buf643, buf644, 4096, 128, grid=grid(4096), stream=stream0)
        buf645 = buf635; del buf635  # reuse
        buf646 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf648 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf642, buf643, buf644, primals_358, primals_359, buf645, buf646, buf648, primals_358, primals_359, 256, 16, grid=grid(256), stream=stream0)
        del primals_358
        del primals_359
        buf649 = reinterpret_tensor(buf640, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf640  # reuse
        buf650 = buf649; del buf649  # reuse
        # Source Nodes: [x_341, x_345], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_77.run(buf650, buf641, buf645, buf646, primals_103, primals_104, 524288, grid=grid(524288), stream=stream0)
        del primals_104
        # Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf652 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_347], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf651, buf652, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf653 = buf644; del buf644  # reuse
        buf654 = buf643; del buf643  # reuse
        buf655 = buf642; del buf642  # reuse
        # Source Nodes: [x_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf652, buf653, buf654, buf655, 4096, 128, grid=grid(4096), stream=stream0)
        buf656 = buf646; del buf646  # reuse
        buf657 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf659 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf653, buf654, buf655, primals_361, primals_362, buf656, buf657, buf659, primals_361, primals_362, 256, 16, grid=grid(256), stream=stream0)
        del primals_361
        del primals_362
        buf661 = reinterpret_tensor(buf651, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf651  # reuse
        buf827 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_348, x_352, xb_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_79.run(buf652, buf656, buf657, primals_105, primals_106, buf639, buf661, buf827, 524288, grid=grid(524288), stream=stream0)
        del primals_106
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf663 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf662, buf663, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf664 = buf655; del buf655  # reuse
        buf665 = buf654; del buf654  # reuse
        buf666 = buf653; del buf653  # reuse
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf663, buf664, buf665, buf666, 4096, 128, grid=grid(4096), stream=stream0)
        buf667 = buf657; del buf657  # reuse
        buf668 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf670 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf664, buf665, buf666, primals_364, primals_365, buf667, buf668, buf670, primals_364, primals_365, 256, 16, grid=grid(256), stream=stream0)
        del primals_364
        del primals_365
        buf671 = reinterpret_tensor(buf662, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf662  # reuse
        buf826 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_355, x_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_80.run(buf663, buf667, buf668, primals_107, primals_108, buf671, buf826, 524288, grid=grid(524288), stream=stream0)
        del buf668
        del primals_108
        buf672 = buf483; del buf483  # reuse
        # Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf484, buf671, buf672, 2048, 512, grid=grid(2048, 512), stream=stream0)
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf674 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf673, buf674, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf675 = buf478; del buf478  # reuse
        buf676 = buf477; del buf477  # reuse
        buf677 = buf476; del buf476  # reuse
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf674, buf675, buf676, buf677, 8192, 128, grid=grid(8192), stream=stream0)
        buf678 = buf480; del buf480  # reuse
        buf679 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf681 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf675, buf676, buf677, primals_367, primals_368, buf678, buf679, buf681, primals_367, primals_368, 512, 16, grid=grid(512), stream=stream0)
        del buf675
        del buf676
        del primals_367
        del primals_368
        buf682 = reinterpret_tensor(buf673, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf673  # reuse
        buf683 = buf682; del buf682  # reuse
        # Source Nodes: [out_3, x_360], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_70.run(buf683, buf674, buf678, buf679, primals_109, primals_110, 1048576, grid=grid(1048576), stream=stream0)
        del primals_110
        # Source Nodes: [x_364], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (8, 1024, 8, 8), (65536, 64, 8, 1))
        buf685 = reinterpret_tensor(buf671, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf671  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf684, buf685, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf686 = reinterpret_tensor(buf666, (1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), 0); del buf666  # reuse
        buf687 = reinterpret_tensor(buf665, (1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), 0); del buf665  # reuse
        buf688 = reinterpret_tensor(buf664, (1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), 0); del buf664  # reuse
        # Source Nodes: [x_365], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf685, buf686, buf687, buf688, 4096, 128, grid=grid(4096), stream=stream0)
        buf689 = reinterpret_tensor(buf65, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf65  # reuse
        buf690 = reinterpret_tensor(buf64, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf64  # reuse
        buf692 = reinterpret_tensor(buf63, (1024, ), (1, ), 0); del buf63  # reuse
        # Source Nodes: [x_365], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_84.run(buf686, buf687, buf688, primals_370, primals_371, buf689, buf690, buf692, primals_370, primals_371, 1024, 4, grid=grid(1024), stream=stream0)
        del primals_370
        del primals_371
        buf693 = reinterpret_tensor(buf684, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf684  # reuse
        buf694 = buf693; del buf693  # reuse
        # Source Nodes: [x_365, x_368], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_85.run(buf694, buf685, buf689, buf690, primals_111, primals_112, 524288, grid=grid(524288), stream=stream0)
        del primals_112
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (8, 1024, 8, 8), (65536, 64, 8, 1))
        buf696 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf695, buf696, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf697 = buf688; del buf688  # reuse
        buf698 = buf687; del buf687  # reuse
        buf699 = buf686; del buf686  # reuse
        # Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf696, buf697, buf698, buf699, 4096, 128, grid=grid(4096), stream=stream0)
        buf700 = buf690; del buf690  # reuse
        buf701 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf703 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_84.run(buf697, buf698, buf699, primals_373, primals_374, buf700, buf701, buf703, primals_373, primals_374, 1024, 4, grid=grid(1024), stream=stream0)
        del primals_373
        del primals_374
        buf704 = reinterpret_tensor(buf695, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf695  # reuse
        # Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_86.run(buf696, buf700, buf701, primals_113, primals_114, buf704, 524288, grid=grid(524288), stream=stream0)
        del primals_114
        buf705 = empty((8, 1024, 8, 8), device='cuda', dtype=torch.float32)
        buf825 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_376], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused_leaky_relu_leaky_relu_backward_87.run(buf704, buf705, buf825, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf706 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf705, buf706, 4096, 64, grid=grid(4096, 64), stream=stream0)
        # Source Nodes: [x_377], Original ATen: [aten.convolution]
        buf707 = extern_kernels.convolution(buf706, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf707, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf708 = buf706; del buf706  # reuse
        # Source Nodes: [x_377], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf707, buf708, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf709 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf710 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf711 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_378], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf708, buf709, buf710, buf711, 2048, 128, grid=grid(2048), stream=stream0)
        buf712 = buf679; del buf679  # reuse
        buf713 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf715 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_378], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf709, buf710, buf711, primals_376, primals_377, buf712, buf713, buf715, primals_376, primals_377, 512, 4, grid=grid(512), stream=stream0)
        del primals_376
        del primals_377
        buf716 = reinterpret_tensor(buf707, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf707  # reuse
        buf717 = buf716; del buf716  # reuse
        # Source Nodes: [x_378, x_382], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92.run(buf717, buf708, buf712, buf713, primals_115, primals_116, 262144, grid=grid(262144), stream=stream0)
        del primals_116
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        buf718 = extern_kernels.convolution(buf717, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf718, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf719 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf718, buf719, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf720 = buf711; del buf711  # reuse
        buf721 = buf710; del buf710  # reuse
        buf722 = buf709; del buf709  # reuse
        # Source Nodes: [x_385], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf719, buf720, buf721, buf722, 2048, 128, grid=grid(2048), stream=stream0)
        buf723 = buf713; del buf713  # reuse
        buf724 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf726 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_385], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf720, buf721, buf722, primals_379, primals_380, buf723, buf724, buf726, primals_379, primals_380, 512, 4, grid=grid(512), stream=stream0)
        del primals_379
        del primals_380
        buf728 = reinterpret_tensor(buf718, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf718  # reuse
        buf824 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_20, x_385, x_389], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_93.run(buf719, buf723, buf724, primals_117, primals_118, buf705, buf728, buf824, 512, 512, grid=grid(512, 512), stream=stream0)
        del primals_118
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf729 = extern_kernels.convolution(buf728, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf729, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf730 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf729, buf730, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf731 = buf722; del buf722  # reuse
        buf732 = buf721; del buf721  # reuse
        buf733 = buf720; del buf720  # reuse
        # Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf730, buf731, buf732, buf733, 2048, 128, grid=grid(2048), stream=stream0)
        buf734 = buf724; del buf724  # reuse
        buf735 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf737 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf731, buf732, buf733, primals_382, primals_383, buf734, buf735, buf737, primals_382, primals_383, 512, 4, grid=grid(512), stream=stream0)
        del primals_382
        del primals_383
        buf738 = reinterpret_tensor(buf729, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf729  # reuse
        buf739 = buf738; del buf738  # reuse
        # Source Nodes: [x_392, x_396], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92.run(buf739, buf730, buf734, buf735, primals_119, primals_120, 262144, grid=grid(262144), stream=stream0)
        del primals_120
        # Source Nodes: [x_398], Original ATen: [aten.convolution]
        buf740 = extern_kernels.convolution(buf739, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf740, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf741 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf740, buf741, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf742 = buf733; del buf733  # reuse
        buf743 = buf732; del buf732  # reuse
        buf744 = buf731; del buf731  # reuse
        # Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf741, buf742, buf743, buf744, 2048, 128, grid=grid(2048), stream=stream0)
        buf745 = buf735; del buf735  # reuse
        buf746 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf748 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf742, buf743, buf744, primals_385, primals_386, buf745, buf746, buf748, primals_385, primals_386, 512, 4, grid=grid(512), stream=stream0)
        del primals_385
        del primals_386
        buf750 = reinterpret_tensor(buf740, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf740  # reuse
        buf823 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_21, x_399, x_403], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_94.run(buf741, buf745, buf746, primals_121, primals_122, buf728, buf750, buf823, 262144, grid=grid(262144), stream=stream0)
        del primals_122
        # Source Nodes: [x_405], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf752 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_405], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf751, buf752, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf753 = buf744; del buf744  # reuse
        buf754 = buf743; del buf743  # reuse
        buf755 = buf742; del buf742  # reuse
        # Source Nodes: [x_406], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf752, buf753, buf754, buf755, 2048, 128, grid=grid(2048), stream=stream0)
        buf756 = buf746; del buf746  # reuse
        buf757 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf759 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_406], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf753, buf754, buf755, primals_388, primals_389, buf756, buf757, buf759, primals_388, primals_389, 512, 4, grid=grid(512), stream=stream0)
        del primals_388
        del primals_389
        buf760 = reinterpret_tensor(buf751, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf751  # reuse
        buf761 = buf760; del buf760  # reuse
        # Source Nodes: [x_406, x_410], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92.run(buf761, buf752, buf756, buf757, primals_123, primals_124, 262144, grid=grid(262144), stream=stream0)
        del primals_124
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf762, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf763 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf762, buf763, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf764 = buf755; del buf755  # reuse
        buf765 = buf754; del buf754  # reuse
        buf766 = buf753; del buf753  # reuse
        # Source Nodes: [x_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf763, buf764, buf765, buf766, 2048, 128, grid=grid(2048), stream=stream0)
        buf767 = buf757; del buf757  # reuse
        buf768 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf770 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf764, buf765, buf766, primals_391, primals_392, buf767, buf768, buf770, primals_391, primals_392, 512, 4, grid=grid(512), stream=stream0)
        del primals_391
        del primals_392
        buf772 = reinterpret_tensor(buf762, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf762  # reuse
        buf822 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut_22, x_413, x_417], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_94.run(buf763, buf767, buf768, primals_125, primals_126, buf750, buf772, buf822, 262144, grid=grid(262144), stream=stream0)
        del primals_126
        # Source Nodes: [x_419], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf773, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf774 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_419], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf773, buf774, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf775 = buf766; del buf766  # reuse
        buf776 = buf765; del buf765  # reuse
        buf777 = buf764; del buf764  # reuse
        # Source Nodes: [x_420], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf774, buf775, buf776, buf777, 2048, 128, grid=grid(2048), stream=stream0)
        buf778 = buf768; del buf768  # reuse
        buf779 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf781 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_420], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf775, buf776, buf777, primals_394, primals_395, buf778, buf779, buf781, primals_394, primals_395, 512, 4, grid=grid(512), stream=stream0)
        del primals_394
        del primals_395
        buf782 = reinterpret_tensor(buf773, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf773  # reuse
        buf783 = buf782; del buf782  # reuse
        # Source Nodes: [x_420, x_424], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_92.run(buf783, buf774, buf778, buf779, primals_127, primals_128, 262144, grid=grid(262144), stream=stream0)
        del primals_128
        # Source Nodes: [x_426], Original ATen: [aten.convolution]
        buf784 = extern_kernels.convolution(buf783, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf784, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf785 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_426], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf784, buf785, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf786 = buf777; del buf777  # reuse
        buf787 = buf776; del buf776  # reuse
        buf788 = buf775; del buf775  # reuse
        # Source Nodes: [x_427], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf785, buf786, buf787, buf788, 2048, 128, grid=grid(2048), stream=stream0)
        buf789 = buf779; del buf779  # reuse
        buf790 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf792 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_427], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf786, buf787, buf788, primals_397, primals_398, buf789, buf790, buf792, primals_397, primals_398, 512, 4, grid=grid(512), stream=stream0)
        del primals_397
        del primals_398
        buf794 = reinterpret_tensor(buf784, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf784  # reuse
        buf821 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_427, x_431, xb_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_leaky_relu_leaky_relu_backward_94.run(buf785, buf789, buf790, primals_129, primals_130, buf772, buf794, buf821, 262144, grid=grid(262144), stream=stream0)
        del primals_130
        # Source Nodes: [x_433], Original ATen: [aten.convolution]
        buf795 = extern_kernels.convolution(buf794, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf795, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf796 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_433], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf795, buf796, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf797 = buf788; del buf788  # reuse
        buf798 = buf787; del buf787  # reuse
        buf799 = buf786; del buf786  # reuse
        # Source Nodes: [x_434], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf796, buf797, buf798, buf799, 2048, 128, grid=grid(2048), stream=stream0)
        buf800 = buf790; del buf790  # reuse
        buf801 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf803 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_434], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf797, buf798, buf799, primals_400, primals_401, buf800, buf801, buf803, primals_400, primals_401, 512, 4, grid=grid(512), stream=stream0)
        del buf797
        del buf798
        del buf799
        del primals_400
        del primals_401
        buf804 = reinterpret_tensor(buf795, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf795  # reuse
        buf820 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_434, x_437], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_95.run(buf796, buf800, buf801, primals_131, primals_132, buf804, buf820, 262144, grid=grid(262144), stream=stream0)
        del buf801
        del primals_132
        buf805 = buf704; del buf704  # reuse
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf705, buf804, buf805, 512, 1024, grid=grid(512, 1024), stream=stream0)
        del buf804
        # Source Nodes: [x_438], Original ATen: [aten.convolution]
        buf806 = extern_kernels.convolution(buf805, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf806, (8, 1024, 8, 8), (65536, 64, 8, 1))
        buf807 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf806, buf807, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf808 = buf699; del buf699  # reuse
        buf809 = buf698; del buf698  # reuse
        buf810 = buf697; del buf697  # reuse
        # Source Nodes: [x_439], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf807, buf808, buf809, buf810, 4096, 128, grid=grid(4096), stream=stream0)
        buf811 = buf701; del buf701  # reuse
        buf812 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf814 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_439], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_84.run(buf808, buf809, buf810, primals_403, primals_404, buf811, buf812, buf814, primals_403, primals_404, 1024, 4, grid=grid(1024), stream=stream0)
        del buf808
        del buf809
        del buf810
        del primals_403
        del primals_404
        buf815 = reinterpret_tensor(buf806, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf806  # reuse
        buf819 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_439, x_444], Original ATen: [aten._native_batch_norm_legit_functional, aten.leaky_relu, aten.leaky_relu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_leaky_relu_backward_97.run(buf807, buf811, buf812, primals_133, primals_134, buf815, buf819, 524288, grid=grid(524288), stream=stream0)
        del buf812
        del primals_134
        buf816 = reinterpret_tensor(buf677, (8, 1024, 1, 1), (1024, 1, 8192, 8192), 0); del buf677  # reuse
        buf817 = reinterpret_tensor(buf816, (8, 1024), (1024, 1), 0); del buf816  # reuse
        # Source Nodes: [x_444, x_445, x_447], Original ATen: [aten.leaky_relu, aten.mean, aten.view]
        triton_per_fused_leaky_relu_mean_view_98.run(buf817, buf815, 8192, 64, grid=grid(8192), stream=stream0)
        del buf815
        buf818 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_449], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_203, buf817, reinterpret_tensor(primals_202, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf818)
        del primals_203
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_204, primals_204, 1, grid=grid(1), stream=stream0)
        del primals_204
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_207, primals_207, 1, grid=grid(1), stream=stream0)
        del primals_207
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_210, primals_210, 1, grid=grid(1), stream=stream0)
        del primals_210
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_213, primals_213, 1, grid=grid(1), stream=stream0)
        del primals_213
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [add__61], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [add__62], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        # Source Nodes: [add__63], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_393, primals_393, 1, grid=grid(1), stream=stream0)
        del primals_393
        # Source Nodes: [add__64], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_396, primals_396, 1, grid=grid(1), stream=stream0)
        del primals_396
        # Source Nodes: [add__65], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_399, primals_399, 1, grid=grid(1), stream=stream0)
        del primals_399
        # Source Nodes: [add__66], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_402, primals_402, 1, grid=grid(1), stream=stream0)
        del primals_402
        return (buf818, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, buf0, buf1, primals_137, primals_138, buf2, primals_140, primals_141, buf3, primals_143, primals_144, buf4, primals_146, buf5, primals_148, primals_149, buf6, primals_151, primals_152, buf7, primals_154, buf8, primals_156, buf9, primals_158, buf10, primals_160, buf11, primals_162, buf12, primals_164, buf13, primals_166, buf14, primals_168, primals_169, buf15, primals_171, primals_172, buf16, primals_174, buf17, primals_176, buf18, primals_178, buf19, primals_180, buf20, primals_182, buf21, primals_184, buf22, primals_186, buf23, primals_188, primals_189, buf24, primals_191, primals_192, buf25, primals_194, buf26, primals_196, buf27, primals_198, buf28, primals_200, primals_201, buf29, buf31, buf41, buf43, buf45, buf55, buf57, buf59, buf69, reinterpret_tensor(buf71, (8, 64, 128, 128), (2097152, 16384, 128, 1), 1048576), buf74, buf84, buf86, buf88, buf98, buf100, buf102, buf112, buf114, buf116, buf126, buf128, buf130, buf140, buf142, buf144, buf154, reinterpret_tensor(buf156, (8, 64, 64, 64), (524288, 4096, 64, 1), 262144), buf159, buf169, buf171, buf173, buf183, buf185, buf187, buf197, buf199, buf201, buf211, buf213, buf215, buf225, buf227, buf229, buf239, buf241, buf243, buf250, buf252, buf254, buf261, reinterpret_tensor(buf263, (8, 128, 32, 32), (262144, 1024, 32, 1), 131072), buf266, buf273, buf275, buf277, buf284, buf286, buf288, buf295, buf297, buf299, buf306, buf308, buf310, buf317, buf319, buf321, buf328, buf330, buf332, buf339, buf341, buf343, buf350, buf352, buf354, buf361, buf363, buf365, buf372, buf374, buf376, buf383, buf385, buf387, buf394, buf396, buf398, buf405, buf407, buf409, buf416, buf418, buf420, buf427, buf429, buf431, buf438, buf440, buf442, buf449, buf451, buf453, buf460, buf462, buf464, buf471, buf473, buf475, buf482, reinterpret_tensor(buf484, (8, 256, 16, 16), (131072, 256, 16, 1), 65536), buf487, buf494, buf496, buf498, buf505, buf507, buf509, buf516, buf518, buf520, buf527, buf529, buf531, buf538, buf540, buf542, buf549, buf551, buf553, buf560, buf562, buf564, buf571, buf573, buf575, buf582, buf584, buf586, buf593, buf595, buf597, buf604, buf606, buf608, buf615, buf617, buf619, buf626, buf628, buf630, buf637, buf639, buf641, buf648, buf650, buf652, buf659, buf661, buf663, buf670, buf672, buf674, buf681, buf683, buf685, buf692, buf694, buf696, buf703, reinterpret_tensor(buf705, (8, 512, 8, 8), (65536, 64, 8, 1), 32768), buf708, buf715, buf717, buf719, buf726, buf728, buf730, buf737, buf739, buf741, buf748, buf750, buf752, buf759, buf761, buf763, buf770, buf772, buf774, buf781, buf783, buf785, buf792, buf794, buf796, buf803, buf805, buf807, buf814, buf817, reinterpret_tensor(primals_202, (1000, 1024), (1024, 1), 0), buf819, reinterpret_tensor(buf811, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf820, reinterpret_tensor(buf800, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf821, reinterpret_tensor(buf789, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf778, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf822, reinterpret_tensor(buf767, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf756, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf823, reinterpret_tensor(buf745, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf734, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf824, reinterpret_tensor(buf723, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf712, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf825, reinterpret_tensor(buf700, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf689, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf678, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf826, reinterpret_tensor(buf667, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf827, reinterpret_tensor(buf656, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf645, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf828, reinterpret_tensor(buf634, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf623, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf829, reinterpret_tensor(buf612, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf601, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf830, reinterpret_tensor(buf590, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf579, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf831, reinterpret_tensor(buf568, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf557, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf832, reinterpret_tensor(buf546, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf535, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf833, reinterpret_tensor(buf524, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf513, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf834, reinterpret_tensor(buf502, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf491, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf835, reinterpret_tensor(buf479, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf468, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf457, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf836, reinterpret_tensor(buf446, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf837, reinterpret_tensor(buf435, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf424, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf838, reinterpret_tensor(buf413, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf402, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf839, reinterpret_tensor(buf391, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf380, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf840, reinterpret_tensor(buf369, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf358, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf841, reinterpret_tensor(buf347, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf336, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf842, reinterpret_tensor(buf325, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf314, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf843, reinterpret_tensor(buf303, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf292, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf844, reinterpret_tensor(buf281, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf270, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf845, reinterpret_tensor(buf258, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf247, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf236, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf846, reinterpret_tensor(buf222, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf847, reinterpret_tensor(buf208, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf194, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf848, reinterpret_tensor(buf180, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf166, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf849, reinterpret_tensor(buf151, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf137, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf123, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf850, reinterpret_tensor(buf109, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf851, reinterpret_tensor(buf95, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf852, reinterpret_tensor(buf66, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf52, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf38, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_367 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_373 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_379 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_382 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_397 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_400 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_403 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cspdarknet53', benchmark_compiled_module)
