
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


# kernel path: /tmp/torchinductor_youkaichao/lb/clbpb4l36uu2k6t3gpw5b5bb7bguuexlzdfmifrsagabm55bko6k.py
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
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ig/cigu27fjbuxis6gnakkkhcl3ijy5xlr7p2dhxozsrebfh46b5aes.py
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czo4ye555ijreyldlwjxr46d5xbmsdlk3ao6zzc3dokazwroftdb.py
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
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3y/c3y5em2irpzhabmj3lora5hxvlxc2q3iktduvc6xxddviobtgdyz.py
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
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vn/cvn6xt5mvyh3pego62alfvjiticdptrnh5ncy7igcsxk3rtpka4n.py
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
    size_hints=[16, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdcpmgd2xkizw4o23icuhdg6i6fnn2m4seevfyuot2qbtvaefus.py
# Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gz/cgzh5bbxiocqcwepf7w6wbqqocv76kn57624krphdhzjpzlapb7a.py
# Source Nodes: [l__mod___features_4], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_4 => getitem, getitem_1
triton_poi_fused_max_pool2d_with_indices_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 112
    x2 = (xindex // 7168)
    x3 = xindex
    x4 = (xindex // 7168) % 112
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (128*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + (128*x1) + (28672*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (448*x4)
    tmp9 = (2*x1) + (448*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 224 + (2*x1) + (448*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 225 + (2*x1) + (448*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwvphtajcnvhrmcqtvpamnibih5pzxnzgctniru7qipmzezrfqs.py
# Source Nodes: [l__mod___features_5, l__mod___features_6], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxfkdzbyqytcrjbs2bnxinngqdrjazfvxqlelryaep77b43aka4.py
# Source Nodes: [l__mod___features_9], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_9 => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168)
    x3 = xindex
    x4 = (xindex // 7168) % 56
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (256*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14464 + x0 + (256*x1) + (28672*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (224*x4)
    tmp9 = (2*x1) + (224*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 112 + (2*x1) + (224*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 113 + (2*x1) + (224*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbohm5hrhixytzjdequ5bag2okezsjyoyj22fuymdbrngsuaj6k.py
# Source Nodes: [l__mod___features_10, l__mod___features_11], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
triton_poi_fused_convolution_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6q5ejh3lzwbytmmtlz3vxg37du3fciayaahwvstopdd7ktpa46.py
# Source Nodes: [l__mod___features_16], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_16 => getitem_4, getitem_5
triton_poi_fused_max_pool2d_with_indices_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168)
    x3 = xindex
    x4 = (xindex // 7168) % 28
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (512*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14592 + x0 + (512*x1) + (28672*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (112*x4)
    tmp9 = (2*x1) + (112*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 56 + (2*x1) + (112*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 57 + (2*x1) + (112*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5lhn2b6ztpabu7q73wzjfpomkujylblqpvar6ng4putcejlobj.py
# Source Nodes: [l__mod___features_17, l__mod___features_18], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
triton_poi_fused_convolution_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ta/ctamfbtmtmplmtdfftprygczn4u2q74irq5wrlphynrhdv4rxejt.py
# Source Nodes: [l__mod___features_23], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_23 => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 14
    x2 = (xindex // 7168)
    x3 = xindex
    x4 = (xindex // 7168) % 14
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (1024*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14848 + x0 + (1024*x1) + (28672*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (56*x4)
    tmp9 = (2*x1) + (56*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 28 + (2*x1) + (56*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 29 + (2*x1) + (56*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbftyxnxzrelatm5dqmf52lq4hhxnj3csitkszwmdj2cyjp73q2.py
# Source Nodes: [l__mod___features_24, l__mod___features_25], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_24 => convolution_10
# l__mod___features_25 => relu_10
triton_poi_fused_convolution_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zc/czcuuy6jeyiwgu5k64zxjm7ydzhmlwr5rogsjtehxxjoqxehamkl.py
# Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# x => getitem_8, getitem_9
triton_poi_fused_max_pool2d_with_indices_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 3584) % 7
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (1024*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + (1024*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (28*x4)
    tmp9 = (2*x1) + (28*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 14 + (2*x1) + (28*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 15 + (2*x1) + (28*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxajuna7bbfbc3kb5un25qier72cspstkptrfb32s5hkhvzclrlm.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._adaptive_avg_pool2d, aten.view]
# x_1 => _adaptive_avg_pool2d
# x_2 => view
triton_poi_fused__adaptive_avg_pool2d_view_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((512*(x0 % 49)) + (25088*x1) + (x0 // 49)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufwob4utsg7dxra436mcbqjccsmneqtqqmlxrbfujd7ckvj2tw7.py
# Source Nodes: [l__mod___classifier_1], Original ATen: [aten.relu, aten.threshold_backward]
# l__mod___classifier_1 => relu_13
triton_poi_fused_relu_threshold_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (4096, 25088), (25088, 1))
    assert_size_stride(primals_28, (4096, ), (1, ))
    assert_size_stride(primals_29, (4096, 4096), (4096, 1))
    assert_size_stride(primals_30, (4096, ), (1, ))
    assert_size_stride(primals_31, (1000, 4096), (4096, 1))
    assert_size_stride(primals_32, (1000, ), (1, ))
    assert_size_stride(primals_33, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_3, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_3
        buf2 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_5, buf2, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_5
        buf3 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_7, buf3, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_7
        buf4 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_9, buf4, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_9
        buf5 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_11, buf5, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_11
        buf6 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_13, buf6, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_13
        buf7 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_15, buf7, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_15
        buf8 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_17, buf8, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_17
        buf9 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_19, buf9, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_19
        buf10 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_21, buf10, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_21
        buf11 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_23, buf11, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_23
        buf12 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_25, buf12, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_25
        buf13 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_33, buf13, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_33
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 224, 224), (3211264, 50176, 224, 1))
        buf15 = empty_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(buf14, primals_2, buf15, 256, 50176, grid=grid(256, 50176), stream=stream0)
        del primals_2
        # Source Nodes: [l__mod___features_2], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 224, 224), (3211264, 50176, 224, 1))
        buf17 = reinterpret_tensor(buf14, (4, 64, 224, 224), (3211264, 1, 14336, 64), 0); del buf14  # reuse
        # Source Nodes: [l__mod___features_2, l__mod___features_3], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(buf16, primals_4, buf17, 256, 50176, grid=grid(256, 50176), stream=stream0)
        del buf16
        del primals_4
        buf18 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        buf19 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_4], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_10.run(buf17, buf18, buf19, 3211264, grid=grid(3211264), stream=stream0)
        # Source Nodes: [l__mod___features_5], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf18, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 128, 112, 112), (1605632, 12544, 112, 1))
        buf21 = empty_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5, l__mod___features_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf20, primals_6, buf21, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del primals_6
        # Source Nodes: [l__mod___features_7], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 128, 112, 112), (1605632, 12544, 112, 1))
        buf23 = reinterpret_tensor(buf20, (4, 128, 112, 112), (1605632, 1, 14336, 128), 0); del buf20  # reuse
        # Source Nodes: [l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf22, primals_8, buf23, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del buf22
        del primals_8
        buf24 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_9], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_12.run(buf23, buf24, buf25, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [l__mod___features_10], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf27 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_10, l__mod___features_11], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_13.run(buf26, primals_10, buf27, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_10
        # Source Nodes: [l__mod___features_12], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf29 = reinterpret_tensor(buf26, (4, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf26  # reuse
        # Source Nodes: [l__mod___features_12, l__mod___features_13], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_13.run(buf28, primals_12, buf29, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_12
        # Source Nodes: [l__mod___features_14], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf31 = reinterpret_tensor(buf28, (4, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf28  # reuse
        # Source Nodes: [l__mod___features_14, l__mod___features_15], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_13.run(buf30, primals_14, buf31, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del buf30
        del primals_14
        buf32 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_16], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_14.run(buf31, buf32, buf33, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [l__mod___features_17], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf35 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_17, l__mod___features_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_15.run(buf34, primals_16, buf35, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_16
        # Source Nodes: [l__mod___features_19], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf37 = reinterpret_tensor(buf34, (4, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf34  # reuse
        # Source Nodes: [l__mod___features_19, l__mod___features_20], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_15.run(buf36, primals_18, buf37, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_18
        # Source Nodes: [l__mod___features_21], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf39 = reinterpret_tensor(buf36, (4, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf36  # reuse
        # Source Nodes: [l__mod___features_21, l__mod___features_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_15.run(buf38, primals_20, buf39, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf38
        del primals_20
        buf40 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_23], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_16.run(buf39, buf40, buf41, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [l__mod___features_24], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf43 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_24, l__mod___features_25], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_17.run(buf42, primals_22, buf43, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del primals_22
        # Source Nodes: [l__mod___features_26], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf45 = reinterpret_tensor(buf42, (4, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf42  # reuse
        # Source Nodes: [l__mod___features_26, l__mod___features_27], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_17.run(buf44, primals_24, buf45, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del primals_24
        # Source Nodes: [l__mod___features_28], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf47 = reinterpret_tensor(buf44, (4, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf44  # reuse
        # Source Nodes: [l__mod___features_28, l__mod___features_29], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_17.run(buf46, primals_26, buf47, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del buf46
        del primals_26
        buf48 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_18.run(buf47, buf48, buf49, 100352, grid=grid(100352), stream=stream0)
        buf50 = empty((4, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._adaptive_avg_pool2d, aten.view]
        triton_poi_fused__adaptive_avg_pool2d_view_19.run(buf48, buf50, 100352, grid=grid(100352), stream=stream0)
        buf51 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf50, reinterpret_tensor(primals_27, (25088, 4096), (1, 25088), 0), out=buf51)
        buf52 = buf51; del buf51  # reuse
        buf57 = empty((4, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_20.run(buf52, primals_28, buf57, 16384, grid=grid(16384), stream=stream0)
        del primals_28
        buf53 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf52, reinterpret_tensor(primals_29, (4096, 4096), (1, 4096), 0), out=buf53)
        buf54 = buf53; del buf53  # reuse
        buf56 = empty((4, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___classifier_4], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_20.run(buf54, primals_30, buf56, 16384, grid=grid(16384), stream=stream0)
        del primals_30
        buf55 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_32, buf54, reinterpret_tensor(primals_31, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf55)
        del primals_32
        return (buf55, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf15, buf17, buf18, buf19, buf21, buf23, buf24, buf25, buf27, buf29, buf31, buf32, buf33, buf35, buf37, buf39, buf40, buf41, buf43, buf45, buf47, buf48, buf49, buf50, buf52, buf54, reinterpret_tensor(primals_31, (1000, 4096), (4096, 1), 0), buf56, reinterpret_tensor(primals_29, (4096, 4096), (4096, 1), 0), buf57, reinterpret_tensor(primals_27, (4096, 25088), (25088, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4096, 25088), (25088, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
