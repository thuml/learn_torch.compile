
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


# kernel path: /tmp/torchinductor_youkaichao/4t/c4ti35dg7x4sl5ammrxfg6cnlcz3tnbd7cwkuu4clpsgx445hlqi.py
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
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu35sx4ulkrbt5hxuk2tonj5rj3yaij3mcrqim6khmk4xlfnyt3o.py
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
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tsny6hxi66qwxrlrhfkan4ysy65es3dvttgtslbevlwew7ikca.py
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


# kernel path: /tmp/torchinductor_youkaichao/rb/crbm5bqw32wdpjkretfjep2fdb5qb56dmazcru7urmg2i6zkei2k.py
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
    size_hints=[4096, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/th/cthu24dz2wgsxgmi74sq4u6xd3yhsivnqhxtyxxt2brjy4ge4lpm.py
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
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ds/cdshl63bky72n5zbzxdnpvtxns5icswtl5vgaivmvocfgsppi2cj.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xhyqoawzp3vbnvwxhn7k3un67tk56eezejavkupjmspnoe32li.py
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
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7fdiqkkscmsfuzinxvhw42577u7uu462h4l2kxzx3anbmlhq45.py
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
    size_hints=[16384, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lo/closylr2wxcwjhegrjhw4nuywlwi75bmljhnddaisw364nmqtpac.py
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
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsmcrlvmddyxfyr7uy2axqey2f4z6lwcmlfmmikb5nh3ryzsglu.py
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
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5gcopnwdldeynixuys6i46vn4j373dc2zg6xuye5y4h3jvboma.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2g/c2g552xypwlos4uw5lpqxhbyoeuromhphud7dtgjbct5kbw6edaf.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/it/citlq2p5cyiytihh3djjahgn64frlt5sxtbhbyz2k6i3vunfvaap.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yq/cyq7zpddvsw7xnvclolfcuvm2ii3hfw6oxlyzxhczjpmcua4dgam.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csxn3uujsihbjbk2oadvjd5vfy5em4qacplbat5rqhj2m7zhlmzq.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dv/cdv2w4aambv4nkz266r57knspxcdpjsog52pj22grl47gjjxf6e6.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6yol74adodaq2gqouvqyv6ygkyaahtvkql4ecxmsqw7wx2s7vj.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6o/c6od4lyamf36md6dr7coglawynloax7dgcyxiz6fh3ozyotudf3a.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (710432*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxyazpp75yygg6jdfl2q5kwb2rnwh54d3ht5du5zac7wvuoxkrpf.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 44416
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 177608, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 177608))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xlmdg76jirb7rspcm663swcn7vwvyon5daoclzc5wsczwcg35e.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 352
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 11
    x1 = (xindex // 11)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x0)
        tmp1 = tl.full([1, 1], 1388, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*r2) + (4064*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (32*r2) + (4064*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (32*r2) + (4064*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwmebcljjaxyzbavlez7ypiza4cuh3ghah6lff6t5ebn3kkclbl.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 11
    RBLOCK: tl.constexpr = 16
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
    tmp16 = 177608.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000056304087113
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


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5prdflov2sb4g2jny3iztjvfvo35rhhm5mq5ovyfzxydjfyz2e.py
# Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_5 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5683456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 177608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7y2q2vxrn46smnfsumku7tmckvhxk5bwvrz4rv3aq2txom62jp.py
# Source Nodes: [x_6], Original ATen: [aten.convolution]
# x_6 => convolution_1
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (691488*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidl4uvhckwv47xpjsa4iyiyvgmqkhce5j2wnaw7gepnhed2yqw5.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43232
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 172872, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 172872))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czigg2qzyjeca3xvupfetduva3qpgrxlwtot5wtxzhvgxnms5bpd.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 352
    rnumel = 123
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 11
    x1 = (xindex // 11)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (123*x0)
        tmp1 = tl.full([1, 1], 1351, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*r2) + (3936*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (32*r2) + (3936*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (32*r2) + (3936*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4zthko73yld2qxy246pdvmih23kml5mv43cp6uztb22kpgj3ka.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 11
    RBLOCK: tl.constexpr = 16
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
    tmp16 = 172872.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000005784660238
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


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7fv4xhnjgxqhvpkf346p7q6ovl5wa2k2365scrgn7tjcrarxus.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_11 => relu_1
# x_7 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5531904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 172872.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuv6u3lrjao22wtobi5nccxuycgakdxyao7kg5wrfmwzzygrg4sn.py
# Source Nodes: [x_12], Original ATen: [aten.convolution]
# x_12 => convolution_2
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 21609
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
    tmp0 = tl.load(in_ptr0 + (x2 + (21609*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1382976*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gcequh6jcjaxz4ydr2abmfetxonjrxnuu6gaxmoji7exf4ouzc.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65856
    rnumel = 168
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (10752*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jtiiao6peopnojd5tmgr4mgegtlxhnbegzerosqtil5777ayke.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 115
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (115*x0)
        tmp1 = tl.full([1, 1], 1029, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*r2) + (7360*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (64*r2) + (7360*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (64*r2) + (7360*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdr44l5nhjtyu5ghvybff3y36jxqjvtqm66h6dc2m2hk2gro2c56.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 9
    RBLOCK: tl.constexpr = 16
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
    tmp16 = 172872.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000005784660238
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


# kernel path: /tmp/torchinductor_youkaichao/am/camcqpl3t7swd7qxaumzpukz4kttwrlryicnnx3plwazdnzfaffz.py
# Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_13 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_17 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11063808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 172872.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4hk4x6q457lggv2sqdvxztbnojzckxslw2xuqzsbfpcjyd3id4.py
# Source Nodes: [x_18], Original ATen: [aten.max_pool2d_with_indices]
# x_18 => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2728448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 73
    x2 = (xindex // 4672) % 73
    x3 = (xindex // 341056)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (9408 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (9472 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (9536 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (18816 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (18880 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (18944 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (294*x2)
    tmp19 = (2*x1) + (294*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (294*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 147 + (2*x1) + (294*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 148 + (2*x1) + (294*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 149 + (2*x1) + (294*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 294 + (2*x1) + (294*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 295 + (2*x1) + (294*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 296 + (2*x1) + (294*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxf6ixcqmzlvl4ff6uj6o5iqw4y4y2l2xyjumg6pbw5yrim2rpsu.py
# Source Nodes: [x_19], Original ATen: [aten.convolution]
# x_19 => convolution_3
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (426320*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhsdcn4vqmuymtmkaeuehci77b3gqggco6ihmv7zqdlil6babvn.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 42632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (128*x1)) % 42632))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corsaarzyjxsomprszgxlitsrct7fvvzpdu2zntiw6sbujdyriip.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (112*x0)
        tmp1 = tl.full([1, 1], 334, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (80*r2) + (8960*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (80*r2) + (8960*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (80*r2) + (8960*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (80*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (80*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (80*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/che2pzrv3zfniwptk4mwhl6vf5qfskp577ur56q4pdua3mqzj2zb.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (80*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 42632.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000234571086768
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


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgjftn4wgpxz5dugacwgdtbndcv7vmusmpduaqnjpszjtcnfow3.py
# Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_20 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_24 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3410560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 42632.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsasdpwftxlsxdmgodocs2v5pib6fwxinnrxg4cbnb7kbvhdmj4.py
# Source Nodes: [x_25], Original ATen: [aten.convolution]
# x_25 => convolution_4
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 5041
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
    tmp0 = tl.load(in_ptr0 + (x2 + (5041*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (967872*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfevcbfi5tyunpr5nzbe7z7n4kafa667ircj36znxp3wk5xbpbu.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 60672
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 40328, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (128*x1)) % 40328))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwf4wj6nzqyw77rvdli3inbohaujra6qqab5rvu5byfik5kmixd.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 106
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (106*x0)
        tmp1 = tl.full([1, 1], 316, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*r2) + (20352*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (192*r2) + (20352*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (192*r2) + (20352*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (192*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (192*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (192*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrh2ixmraufuztfizxcw2rj57ca26zg46f4hef5vv5tgvywnaua.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => add_21, add_22, add_23, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 40328.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000247972822178
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


# kernel path: /tmp/torchinductor_youkaichao/av/cav66z5g4i62gaagg4n6ltj4rhzqoygbpyco2bnixp7nk5fcaliy.py
# Source Nodes: [x_26, x_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_26 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_30 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7742976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 40328.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybopq72s5njgu23zsyaqi64ep42nhfineylanton7vqhfajvaxo.py
# Source Nodes: [x_31], Original ATen: [aten.max_pool2d_with_indices]
# x_31 => getitem_12, getitem_13
triton_poi_fused_max_pool2d_with_indices_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192) % 35
    x2 = (xindex // 6720) % 35
    x3 = (xindex // 235200)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (13632 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (13824 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (14016 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (27264 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (27456 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (27648 + x0 + (384*x1) + (27264*x2) + (967872*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (142*x2)
    tmp19 = (2*x1) + (142*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (142*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 71 + (2*x1) + (142*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 72 + (2*x1) + (142*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 73 + (2*x1) + (142*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 142 + (2*x1) + (142*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 143 + (2*x1) + (142*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 144 + (2*x1) + (142*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkgppeng6rx2tzbrwbhjcijjs2ksnqv5b7ihosogfwakcovctc4.py
# Source Nodes: [x_32], Original ATen: [aten.convolution]
# x_32 => convolution_5
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (78400*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3a4qlkom4dpxjmtatfr3kqpim7tbonacuoabr4kofjzszgfgeaf.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvluncqcbczst3llbvlwzuuhb7cfov262x4x3uf7giq3ax47ih77.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_red_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 9800.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001020512297174
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkgekiq5sbr6xejzvxgcos44a4qlsj2v75fk7xcxi2ezmb53yca.py
# Source Nodes: [branch1x1, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1 => relu_5
# x_33 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (78400*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cyp5kouzo4jdlza74lshlleijgi6wv3bwaaj5xmenpqprwwqpano.py
# Source Nodes: [x_37], Original ATen: [aten.convolution]
# x_37 => convolution_6
triton_poi_fused_convolution_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (48*x2) + (58800*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/csci4qolzucuspykxbqkx425lsomnusmn5teuk742htx6nb4sv7x.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3696
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32bbda3ghgscqsguyijdqo2wfhgz3hutx4n3sdpxox5kvkkzgnv.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => add_31, add_32, add_33, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_19, var_mean_6
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 9800.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001020512297174
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbq46clxrlbz6yuwm4eo3rboulg3ib4a2itk5ttlu66ensbxrs4e.py
# Source Nodes: [branch5x5, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch5x5 => relu_6
# x_38 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
triton_poi_fused__native_batch_norm_legit_functional_relu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 470400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxaea6kjx7gfowbeqw2tewqnszrkh672sseb3bxhwu4ddb5lz5i.py
# Source Nodes: [branch3x3dbl, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl => relu_8
# x_48 => add_41, add_44, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_relu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lxs7pr2s7bpq7xlw2sfm6o6xgngnmxzqzyh4gjzvjrggugw7pc.py
# Source Nodes: [x_52], Original ATen: [aten.convolution]
# x_52 => convolution_9
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (117600*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5msctnrkdfrtioj7yg7qchww74lefsjsvbbdhbrs5w72sqtrte.py
# Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
# x_53 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtgi27q7dxtb2cy5ayp3rc5wqgtvjmoqpkail46ebobkuuzh6ro.py
# Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
# x_53 => add_46, add_47, add_48, mul_64, mul_65, mul_66, mul_67, mul_68, rsqrt_9, squeeze_28, var_mean_9
triton_red_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 9800.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001020512297174
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswfzewwgjmlyrnw2pcrbd4nzm2w553momzztbfu46orerm5nb55.py
# Source Nodes: [branch3x3dbl_1, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl_1 => relu_9
# x_53 => add_46, add_49, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_relu_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/creqh6hanguqgcdfv64wexsqb5gen464xit74f2gwumrpd7zcybv.py
# Source Nodes: [branch3x3dbl_2, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_2 => relu_10
# x_58 => add_51, add_54, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (117600*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4ihzewm556zzlrwgh5pddwxfrhwzzh6tqodfggnyqzonc5zyog.py
# Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# branch_pool => avg_pool2d
triton_poi_fused_avg_pool2d_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_59', 'mutated_arg_names': []},
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
    x3 = (xindex // 35)
    x2 = xindex % 35
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6912) + y0 + (192*x5) + (235200*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-6720) + y0 + (192*x5) + (235200*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x2
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6528) + y0 + (192*x5) + (235200*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-192) + y0 + (192*x5) + (235200*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (192 + y0 + (192*x5) + (235200*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x3
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6528 + y0 + (192*x5) + (235200*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (6720 + y0 + (192*x5) + (235200*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6912 + y0 + (192*x5) + (235200*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + x3, [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + x2, [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + x2, [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + x3, [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hdmco4ur2h666fjdpdv7yb3xxsmjnmmv5yru3ol2n5y7vw7kvn.py
# Source Nodes: [x_62], Original ATen: [aten.convolution]
# x_62 => convolution_11
triton_poi_fused_convolution_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (39200*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbt5y575u3hzbzxrlscxdw24vyjukpzjcj3svftpvt66amu3xcg.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7p/c7p52t52wwbjjteej5lxotsejqfgr3jg25co6xp2xz2qkrmmumwx.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => add_56, add_57, add_58, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 9800.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001020512297174
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsakuveudmavdt2hehinsnabvstjlfzwxr67kl67insdklojk3y.py
# Source Nodes: [branch_pool_1, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch_pool_1 => relu_11
# x_63 => add_56, add_59, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (32*x2) + (39200*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevh6trv4wv2clnblnxf6ayxtyzmlkhjs62mpdt44dbes4zsubxb.py
# Source Nodes: [cat_29], Original ATen: [aten.cat]
# cat_29 => cat
triton_poi_fused_cat_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (313600*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbzwabpb4liykvomdsptajju3ebe7uisnvdl634jo3nwx2vpcbl.py
# Source Nodes: [branch1x1_1, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_1 => relu_12
# x_69 => add_61, add_64, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (352800*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (78400*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2au6phfz7yezc42ivfc6rt2w7xoc5cassqd3uxcr4enqjhqb5qc.py
# Source Nodes: [branch3x3dbl_5, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_5 => relu_17
# x_94 => add_86, add_89, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (352800*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (117600*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr34hwk2fbkfcsvgf4rj3n3fjao2bosxr6esixs7y4zurjcdqdbb.py
# Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# branch_pool_2 => avg_pool2d_1
triton_poi_fused_avg_pool2d_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_67', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumytms7uc6cozby56yek23bv6k7avhzp6r6hkqe3m6z66yxs5z5.py
# Source Nodes: [cat_28], Original ATen: [aten.cat]
# cat_28 => cat_1
triton_poi_fused_cat_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (288*x2) + (352800*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capzio7qhcqvxz4iacgo6ds4xirmk7pp7wqiiriodehecohuscmr.py
# Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# branch_pool_4 => avg_pool2d_2
triton_poi_fused_avg_pool2d_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_69', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ysfg7cpm5agc67fodi2nf44fsj6ton6ygasiijkdtz2kg2j7cp.py
# Source Nodes: [x_140], Original ATen: [aten.convolution]
# x_140 => convolution_26
triton_poi_fused_convolution_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 289
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (110976*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/csc4khhohr7rq3io6khgfd3cxmkpdied2dlwuyicueoafo4bdshv.py
# Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
# x_141 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7296
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnm5xeawmugatlwoeh3q3ckiy5rroqqq4xl4sf2mttdq4s6xhctt.py
# Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
# x_141 => add_131, add_132, add_133, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, squeeze_79, var_mean_26
triton_per_fused__native_batch_norm_legit_functional_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_72', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004327131112072
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


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4vushgtsbx6bt5tyweyk5m6rgm567kjfp5bluekdp3lq62wz3i.py
# Source Nodes: [branch3x3, x_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3 => relu_26
# x_141 => add_131, add_134, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (110976*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (384*x2) + (110976*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct56w6sfvqqkr4q53qiryhehoi37nv3owr3vfg3cvalwiikbfj7m.py
# Source Nodes: [x_155], Original ATen: [aten.convolution]
# x_155 => convolution_29
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 289
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (27744*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdxhalcf2bglb6djo5wclx253fewnkvdboak2djdoa4dhefmt3u.py
# Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
# x_156 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1824
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2aeqjtysvisubvf4hab347qliyq3r2pozub7otgexilg7vbbrk5.py
# Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
# x_156 => add_146, add_147, add_148, mul_204, mul_205, mul_206, mul_207, mul_208, rsqrt_29, squeeze_88, var_mean_29
triton_per_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004327131112072
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3o7q75bsdbfn4fnwcs6fc7dagk7sh62lfkeg2esldzps4ig3av.py
# Source Nodes: [branch3x3dbl_11, x_156], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_11 => relu_29
# x_156 => add_146, add_149, mul_203, mul_209, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (27744*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (27744*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5l3hzsa2cyb3jwdfkfcgtpp7ljozbcamblwndd6dm4r7o5xe36m.py
# Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_6 => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_78', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabcccmybba34uvrrvwm5mqmwffg3fc3hbr46brqfnoxuoeqnsgc.py
# Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_6 => getitem_65
triton_poi_fused_max_pool2d_with_indices_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 665856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 288
    x1 = (xindex // 288) % 17
    x2 = (xindex // 4896) % 17
    x3 = (xindex // 83232)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (288 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (576 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp12 = tl.load(in_ptr0 + (10080 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp17 = tl.load(in_ptr0 + (10368 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp22 = tl.load(in_ptr0 + (10656 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp27 = tl.load(in_ptr0 + (20160 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp32 = tl.load(in_ptr0 + (20448 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp37 = tl.load(in_ptr0 + (20736 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = 1 + (2*x1) + (70*x2)
    tmp4 = (2*x1) + (70*x2)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = 2 + (2*x1) + (70*x2)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = 35 + (2*x1) + (70*x2)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = 36 + (2*x1) + (70*x2)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = 37 + (2*x1) + (70*x2)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tmp28 = tmp27 > tmp26
    tmp29 = 70 + (2*x1) + (70*x2)
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp31 = triton_helpers.maximum(tmp27, tmp26)
    tmp33 = tmp32 > tmp31
    tmp34 = 71 + (2*x1) + (70*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp36 = triton_helpers.maximum(tmp32, tmp31)
    tmp38 = tmp37 > tmp36
    tmp39 = 72 + (2*x1) + (70*x2)
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = triton_helpers.maximum(tmp37, tmp36)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2ettpnctimrybm7xiyvjcotddsfdw4mbpdwc6fuw4fyjxb2nhqr.py
# Source Nodes: [cat_26], Original ATen: [aten.cat]
# cat_26 => cat_3
triton_poi_fused_cat_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cip7mggr3r2aizr7orpta2ew5aabs6ebq32in2hgamzs4dmd4a3r.py
# Source Nodes: [x_161], Original ATen: [aten.convolution]
# x_161 => convolution_30
triton_poi_fused_convolution_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (55488*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3nlxufddhdlcapozbn7omqd7khrdzg5d7lbnsaz3wc4kpxqnno.py
# Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
# x_162 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedhrzatkqbrd22wsyxqek43c26twwgwrwsalcmhf4grccttp7ue.py
# Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
# x_162 => add_151, add_152, add_153, mul_211, mul_212, mul_213, mul_214, mul_215, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004327131112072
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


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsloqzcqp4vi6jg4vmaagw7rwe27tau7xw4nd4cqvyaqkc56a66.py
# Source Nodes: [branch1x1_3, x_162], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_3 => relu_30
# x_162 => add_151, add_154, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (55488*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqat2scfcl5nfqhpil7wylseyohvkorxl2eminyb3o7moahbebbu.py
# Source Nodes: [x_166], Original ATen: [aten.convolution]
# x_166 => convolution_31
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (36992*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4k5cmlhyprkk3rbeupveywx35d74kq6itf7bugnzvadm5pfvfgk.py
# Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
# x_167 => var_mean_31
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pnrr74p2lbeglnqmexjvb7cgxgcd7zjnqv4bknxyvcehrr6gud.py
# Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
# x_167 => add_156, add_157, add_158, mul_218, mul_219, mul_220, mul_221, mul_222, rsqrt_31, squeeze_94, var_mean_31
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 19
    RBLOCK: tl.constexpr = 32
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
    tmp16 = 2312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004327131112072
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwtupfv4pypxaszjqin7x4cfkxurb4qb4qsfei5ntgkbcfvrjkg.py
# Source Nodes: [branch7x7, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7 => relu_31
# x_167 => add_156, add_159, mul_217, mul_223, rsqrt_31, sub_31, var_mean_31
triton_poi_fused__native_batch_norm_legit_functional_relu_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2zvrwrgmwdob4mh56qjqoetvhkbdmazxycm253y5wxu4a2n652.py
# Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
# branch_pool_7 => avg_pool2d_3
triton_poi_fused_avg_pool2d_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_89', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/np/cnpe4rj5pyckfm3husydzm7orw2xvctbxqwpr4kmokikyzikxxss.py
# Source Nodes: [x_217], Original ATen: [aten.convolution]
# x_217 => convolution_41
triton_poi_fused_convolution_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (160*x2) + (46240*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvlq6w65p42kwqg3drzuzcjkgabr4poy7v2eqocj3asr5ktjexq.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
# x_218 => var_mean_41
triton_red_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3040
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 160)
    x0 = xindex % 160
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (160*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnu2hbek5grxvlury3g767izn3qhlgxnvt44debh73bfyjino36.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
# x_218 => add_206, add_207, add_208, mul_288, mul_289, mul_290, mul_291, mul_292, rsqrt_41, squeeze_124, var_mean_41
triton_per_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (160*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004327131112072
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


# kernel path: /tmp/torchinductor_youkaichao/tg/ctg4f4jrbi5e5hjgvbaj57mnz32ywgnqgmpflakigkrzq5y342oa.py
# Source Nodes: [branch7x7_3, x_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7_3 => relu_41
# x_218 => add_206, add_209, mul_287, mul_293, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_relu_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 369920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67kdvmteolymu64ynyto6sfyxfpk7dn37mjman5omp6lw5jho6p.py
# Source Nodes: [branch7x7_9, x_320], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7_9 => relu_61
# x_320 => add_306, add_309, mul_427, mul_433, rsqrt_61, sub_61, var_mean_61
triton_poi_fused__native_batch_norm_legit_functional_relu_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 443904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7hwknz2wltgt4gd7764scagyke2m6mhawmcolfs66rsdi4zipo.py
# Source Nodes: [x_371], Original ATen: [aten.convolution]
# x_371 => convolution_71
triton_poi_fused_convolution_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (20480*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosgkg5ve2w3hopyhf5py65qjmrtdu74izjuvbvocnfpmvu3qtyz.py
# Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
# x_372 => var_mean_71
triton_red_fused__native_batch_norm_legit_functional_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdsldev34wryitsfya3p5icnzxk4xnah7txwstfo7knjerhr5tn.py
# Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
# x_372 => add_356, add_357, add_358, mul_498, mul_499, mul_500, mul_501, mul_502, rsqrt_71, squeeze_214, var_mean_71
triton_per_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (320*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ip/cip2kdwce2a5ld7fsirfoirvvr36bbxm4vaojxjfdl25sphmd7qp.py
# Source Nodes: [branch3x3_2, x_372], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3_2 => relu_71
# x_372 => add_356, add_359, mul_497, mul_503, rsqrt_71, sub_71, var_mean_71
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (20480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (81920*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (320*x2) + (20480*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnpiakwcyb45am5r4lj3igjesj4ni7bnjmkcutvl23znavvpkspw.py
# Source Nodes: [x_391], Original ATen: [aten.convolution]
# x_391 => convolution_75
triton_poi_fused_convolution_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (12288*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpcqfpkzjtv6afhyawcepopc2ppw3gs4dest3dqmumkmsslgqlu.py
# Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
# x_392 => var_mean_75
triton_red_fused__native_batch_norm_legit_functional_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnvpvbok7penhupaytqgph32emhxouoxzw567ni6ydh77qzyqy7.py
# Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
# x_392 => add_376, add_377, add_378, mul_526, mul_527, mul_528, mul_529, mul_530, rsqrt_75, squeeze_226, var_mean_75
triton_per_fused__native_batch_norm_legit_functional_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_101', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csf2i7fdshrkqac344xxoz7mkzmnur5rq45r5wfflatr3g7icns4.py
# Source Nodes: [branch7x7x3_3, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch7x7x3_3 => relu_75
# x_392 => add_376, add_379, mul_525, mul_531, rsqrt_75, sub_75, var_mean_75
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (12288*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (81920*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (12288*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbixyyg37e4uezml3vo7frdv4u7jgduljbrmpyw6sxl7rfatcqzq.py
# Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_15 => max_pool2d_with_indices_3
triton_poi_fused_max_pool2d_with_indices_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_103', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyo7kvzejdfrzcqbcwqzbyde4o6wgudv6mmlccwqazzpsy7u4um.py
# Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_15 => getitem_159
triton_poi_fused_max_pool2d_with_indices_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 8
    x2 = (xindex // 6144) % 8
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp7 = tl.load(in_ptr0 + (1536 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp12 = tl.load(in_ptr0 + (13056 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp17 = tl.load(in_ptr0 + (13824 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp22 = tl.load(in_ptr0 + (14592 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp27 = tl.load(in_ptr0 + (26112 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp32 = tl.load(in_ptr0 + (26880 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp37 = tl.load(in_ptr0 + (27648 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp2 = tmp1 > tmp0
    tmp3 = 1 + (2*x1) + (34*x2)
    tmp4 = (2*x1) + (34*x2)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = 2 + (2*x1) + (34*x2)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = 17 + (2*x1) + (34*x2)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = 18 + (2*x1) + (34*x2)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = 19 + (2*x1) + (34*x2)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tmp28 = tmp27 > tmp26
    tmp29 = 34 + (2*x1) + (34*x2)
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp31 = triton_helpers.maximum(tmp27, tmp26)
    tmp33 = tmp32 > tmp31
    tmp34 = 35 + (2*x1) + (34*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp36 = triton_helpers.maximum(tmp32, tmp31)
    tmp38 = tmp37 > tmp36
    tmp39 = 36 + (2*x1) + (34*x2)
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = triton_helpers.maximum(tmp37, tmp36)
    tl.store(out_ptr0 + (x4), tmp40, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cecsyeeubehjzmyagqjei247niwywzmldtndtgkmdzrumi663q2v.py
# Source Nodes: [cat_21], Original ATen: [aten.cat]
# cat_21 => cat_8
triton_poi_fused_cat_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvogn4jdupflm5vd52i7kmvy2boq2xu6ggrkgublttuyq2on2n7.py
# Source Nodes: [branch1x1_7, x_398], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_7 => relu_76
# x_398 => add_381, add_384, mul_532, mul_538, rsqrt_76, sub_76, var_mean_76
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (20480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (131072*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (320*x2) + (20480*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgtmznjzhosn6eaxsgk6uqshzypkschj7ujro2svvk7lhxgkeaw.py
# Source Nodes: [x_402], Original ATen: [aten.convolution]
# x_402 => convolution_77
triton_poi_fused_convolution_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (384*x2) + (24576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnjcqcvi3k424ekh6qlxyo6s77hyq5j7sl4jpqulckiepyy2t2z.py
# Source Nodes: [x_403], Original ATen: [aten._native_batch_norm_legit_functional]
# x_403 => var_mean_77
triton_red_fused__native_batch_norm_legit_functional_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj4rh6wzyz66x5ney5m4iblebqaz3hk2q65owzd3wj2sm32sd7a7.py
# Source Nodes: [x_403], Original ATen: [aten._native_batch_norm_legit_functional]
# x_403 => add_386, add_387, add_388, mul_540, mul_541, mul_542, mul_543, mul_544, rsqrt_77, squeeze_232, var_mean_77
triton_per_fused__native_batch_norm_legit_functional_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_109', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxa7lf37pfficrfn3e6ziw2dc5o67mdxpph2tts2mgaoznbsxron.py
# Source Nodes: [branch3x3_3, x_403], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3_3 => relu_77
# x_403 => add_386, add_389, mul_539, mul_545, rsqrt_77, sub_77, var_mean_77
triton_poi_fused__native_batch_norm_legit_functional_relu_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumnkbxe3fepscrnqmvh7kq2ru4dlgti2ba3je3mic4g4duojffh.py
# Source Nodes: [x_408, x_411], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_408 => add_391, add_394, mul_546, mul_552, rsqrt_78, sub_78, var_mean_78
# x_411 => relu_78
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (49152*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (384*x2) + (24576*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cys7kbdxmm53pv76k7mpn3boinae3buhxu5gxtjtt6pfoj3iwyrh.py
# Source Nodes: [x_417], Original ATen: [aten.convolution]
# x_417 => convolution_80
triton_poi_fused_convolution_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (448*x2) + (28672*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v4/cv46hlvki4natnc6e6j4b2anx5faznjvxblnhtyllmft2raxocrz.py
# Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
# x_418 => var_mean_80
triton_red_fused__native_batch_norm_legit_functional_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1792
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 448
    x1 = (xindex // 448)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cde2sf2yhnqpr4jjeym43iqntdgk7z35bdc6lmwl4lxqtjtoklp7.py
# Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
# x_418 => add_401, add_402, add_403, mul_561, mul_562, mul_563, mul_564, mul_565, rsqrt_80, squeeze_241, var_mean_80
triton_per_fused__native_batch_norm_legit_functional_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_114', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (448*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ro/croujgqjg3igogostuvey43cnqrvitgdslsdwwveg5vsbroqspo7.py
# Source Nodes: [branch3x3dbl_12, x_418], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl_12 => relu_80
# x_418 => add_401, add_404, mul_560, mul_566, rsqrt_80, sub_80, var_mean_80
triton_poi_fused__native_batch_norm_legit_functional_relu_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46jk7wafibcyturqqnxm67enj4nwqlooyjtuqgxdhygg6fm56mg.py
# Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
# branch_pool_16 => avg_pool2d_7
triton_poi_fused_avg_pool2d_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_116', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10240) % 8
    x1 = (xindex // 1280) % 8
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11520) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-10240) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-8960) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1280) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1280 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (8960 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (10240 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (11520 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 9, tl.int64)
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


# kernel path: /tmp/torchinductor_youkaichao/bo/cbok4bkdzzxo2vo2mifcqir7y3p3vtr3uxbmqmo36p2t3a4fcs5b.py
# Source Nodes: [branch_pool_17, x_438], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch_pool_17 => relu_84
# x_438 => add_421, add_424, mul_588, mul_594, rsqrt_84, sub_84, var_mean_84
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (12288*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (131072*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (12288*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catmlot6quu6upsizpez3a4h3xxltdypfbweclsfbnj7kvltaok4.py
# Source Nodes: [cat_18], Original ATen: [aten.cat]
# cat_18 => cat_11
triton_poi_fused_cat_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 49152
    x1 = (xindex // 49152)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (131072*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cda2bgs4m2gxanob5yv3zya3urybfezfklgvbm7kinum7525x3bk.py
# Source Nodes: [cat_18], Original ATen: [aten.cat]
# cat_18 => cat_11
triton_poi_fused_cat_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjvchknuw5ozdonzwjdxb2jr5l2hbwjqc4nk3epgmwgjmc2cf5z.py
# Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
# branch_pool_18 => avg_pool2d_8
triton_poi_fused_avg_pool2d_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16384) % 8
    x1 = (xindex // 2048) % 8
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-18432) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16384) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-14336) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-2048) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (2048 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (14336 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (16384 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (18432 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 9, tl.int64)
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


# kernel path: /tmp/torchinductor_youkaichao/hs/chssi53qdgk77zqrp7nraofhduphrn5edi6sbjl3qmdfq2fwryel.py
# Source Nodes: [x_491, x_493], Original ATen: [aten.mean, aten.view]
# x_491 => mean
# x_493 => view
triton_per_fused_mean_view_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_121', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/sd/csdcykvlzptds2kvumc46hxrqzu6ndibtss2ivxalpec54pnuvze.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_122', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (80, ), (1, ))
    assert_size_stride(primals_8, (80, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (48, ), (1, ))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_42, (48, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (96, ), (1, ))
    assert_size_stride(primals_48, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (96, ), (1, ))
    assert_size_stride(primals_58, (96, ), (1, ))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_60, (96, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (192, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_68, (192, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_78, (192, ), (1, ))
    assert_size_stride(primals_79, (192, ), (1, ))
    assert_size_stride(primals_80, (192, ), (1, ))
    assert_size_stride(primals_81, (192, ), (1, ))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (160, ), (1, ))
    assert_size_stride(primals_85, (160, ), (1, ))
    assert_size_stride(primals_86, (160, ), (1, ))
    assert_size_stride(primals_87, (192, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_90, (160, ), (1, ))
    assert_size_stride(primals_91, (160, ), (1, ))
    assert_size_stride(primals_92, (160, ), (1, ))
    assert_size_stride(primals_93, (160, ), (1, ))
    assert_size_stride(primals_94, (160, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_96, (160, ), (1, ))
    assert_size_stride(primals_97, (192, ), (1, ))
    assert_size_stride(primals_98, (192, ), (1, ))
    assert_size_stride(primals_99, (192, ), (1, ))
    assert_size_stride(primals_100, (192, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_102, (192, ), (1, ))
    assert_size_stride(primals_103, (160, ), (1, ))
    assert_size_stride(primals_104, (160, ), (1, ))
    assert_size_stride(primals_105, (160, ), (1, ))
    assert_size_stride(primals_106, (160, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_108, (192, ), (1, ))
    assert_size_stride(primals_109, (160, ), (1, ))
    assert_size_stride(primals_110, (160, ), (1, ))
    assert_size_stride(primals_111, (160, ), (1, ))
    assert_size_stride(primals_112, (160, ), (1, ))
    assert_size_stride(primals_113, (160, ), (1, ))
    assert_size_stride(primals_114, (160, ), (1, ))
    assert_size_stride(primals_115, (160, ), (1, ))
    assert_size_stride(primals_116, (160, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (192, ), (1, ))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_120, (192, ), (1, ))
    assert_size_stride(primals_121, (192, ), (1, ))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_124, (192, ), (1, ))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_126, (192, ), (1, ))
    assert_size_stride(primals_127, (192, ), (1, ))
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_130, (192, ), (1, ))
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_132, (192, ), (1, ))
    assert_size_stride(primals_133, (192, ), (1, ))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (192, ), (1, ))
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_138, (192, ), (1, ))
    assert_size_stride(primals_139, (192, ), (1, ))
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (192, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_144, (320, ), (1, ))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (192, ), (1, ))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (320, ), (1, ))
    assert_size_stride(primals_154, (320, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (448, ), (1, ))
    assert_size_stride(primals_162, (448, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (320, ), (1, ))
    assert_size_stride(primals_172, (320, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (448, ), (1, ))
    assert_size_stride(primals_180, (448, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_186, (384, ), (1, ))
    assert_size_stride(primals_187, (192, ), (1, ))
    assert_size_stride(primals_188, (192, ), (1, ))
    assert_size_stride(primals_189, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_190, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_191, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_192, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_193, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_194, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_195, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_196, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_197, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_199, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_200, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_201, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_204, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_205, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_206, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_207, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_208, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_209, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_210, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_211, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_212, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_213, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_214, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_215, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_216, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_217, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_218, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_219, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_221, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_222, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_223, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_225, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_226, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_227, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_228, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_229, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_230, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_231, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_232, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_233, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_234, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_235, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_236, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_237, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_238, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_239, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_240, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_241, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_242, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_243, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_244, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_245, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_246, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_247, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_248, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_249, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_252, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_253, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_254, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_255, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_256, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_257, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_258, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_260, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_261, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_262, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_263, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_264, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_265, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_266, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_267, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_268, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_269, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_271, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_272, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_273, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_274, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_275, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_276, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_277, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_278, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_279, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_280, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_281, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_282, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_283, (1000, 2048), (2048, 1))
    assert_size_stride(primals_284, (1000, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (32, ), (1, ))
    assert_size_stride(primals_287, (32, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (32, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (80, ), (1, ))
    assert_size_stride(primals_296, (80, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (192, ), (1, ))
    assert_size_stride(primals_299, (192, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (48, ), (1, ))
    assert_size_stride(primals_305, (48, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (64, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (96, ), (1, ))
    assert_size_stride(primals_314, (96, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (96, ), (1, ))
    assert_size_stride(primals_317, (96, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (32, ), (1, ))
    assert_size_stride(primals_320, (32, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (64, ), (1, ))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (48, ), (1, ))
    assert_size_stride(primals_326, (48, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (64, ), (1, ))
    assert_size_stride(primals_329, (64, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (64, ), (1, ))
    assert_size_stride(primals_332, (64, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (96, ), (1, ))
    assert_size_stride(primals_335, (96, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (96, ), (1, ))
    assert_size_stride(primals_338, (96, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (48, ), (1, ))
    assert_size_stride(primals_347, (48, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (64, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (64, ), (1, ))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (96, ), (1, ))
    assert_size_stride(primals_356, (96, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (96, ), (1, ))
    assert_size_stride(primals_359, (96, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (64, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (384, ), (1, ))
    assert_size_stride(primals_365, (384, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (96, ), (1, ))
    assert_size_stride(primals_371, (96, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (96, ), (1, ))
    assert_size_stride(primals_374, (96, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (192, ), (1, ))
    assert_size_stride(primals_377, (192, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (128, ), (1, ))
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (192, ), (1, ))
    assert_size_stride(primals_386, (192, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (128, ), (1, ))
    assert_size_stride(primals_389, (128, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (128, ), (1, ))
    assert_size_stride(primals_392, (128, ), (1, ))
    assert_size_stride(primals_393, (), ())
    assert_size_stride(primals_394, (128, ), (1, ))
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (), ())
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (128, ), (1, ))
    assert_size_stride(primals_399, (), ())
    assert_size_stride(primals_400, (192, ), (1, ))
    assert_size_stride(primals_401, (192, ), (1, ))
    assert_size_stride(primals_402, (), ())
    assert_size_stride(primals_403, (192, ), (1, ))
    assert_size_stride(primals_404, (192, ), (1, ))
    assert_size_stride(primals_405, (), ())
    assert_size_stride(primals_406, (192, ), (1, ))
    assert_size_stride(primals_407, (192, ), (1, ))
    assert_size_stride(primals_408, (), ())
    assert_size_stride(primals_409, (160, ), (1, ))
    assert_size_stride(primals_410, (160, ), (1, ))
    assert_size_stride(primals_411, (), ())
    assert_size_stride(primals_412, (160, ), (1, ))
    assert_size_stride(primals_413, (160, ), (1, ))
    assert_size_stride(primals_414, (), ())
    assert_size_stride(primals_415, (192, ), (1, ))
    assert_size_stride(primals_416, (192, ), (1, ))
    assert_size_stride(primals_417, (), ())
    assert_size_stride(primals_418, (160, ), (1, ))
    assert_size_stride(primals_419, (160, ), (1, ))
    assert_size_stride(primals_420, (), ())
    assert_size_stride(primals_421, (160, ), (1, ))
    assert_size_stride(primals_422, (160, ), (1, ))
    assert_size_stride(primals_423, (), ())
    assert_size_stride(primals_424, (160, ), (1, ))
    assert_size_stride(primals_425, (160, ), (1, ))
    assert_size_stride(primals_426, (), ())
    assert_size_stride(primals_427, (160, ), (1, ))
    assert_size_stride(primals_428, (160, ), (1, ))
    assert_size_stride(primals_429, (), ())
    assert_size_stride(primals_430, (192, ), (1, ))
    assert_size_stride(primals_431, (192, ), (1, ))
    assert_size_stride(primals_432, (), ())
    assert_size_stride(primals_433, (192, ), (1, ))
    assert_size_stride(primals_434, (192, ), (1, ))
    assert_size_stride(primals_435, (), ())
    assert_size_stride(primals_436, (192, ), (1, ))
    assert_size_stride(primals_437, (192, ), (1, ))
    assert_size_stride(primals_438, (), ())
    assert_size_stride(primals_439, (160, ), (1, ))
    assert_size_stride(primals_440, (160, ), (1, ))
    assert_size_stride(primals_441, (), ())
    assert_size_stride(primals_442, (160, ), (1, ))
    assert_size_stride(primals_443, (160, ), (1, ))
    assert_size_stride(primals_444, (), ())
    assert_size_stride(primals_445, (192, ), (1, ))
    assert_size_stride(primals_446, (192, ), (1, ))
    assert_size_stride(primals_447, (), ())
    assert_size_stride(primals_448, (160, ), (1, ))
    assert_size_stride(primals_449, (160, ), (1, ))
    assert_size_stride(primals_450, (), ())
    assert_size_stride(primals_451, (160, ), (1, ))
    assert_size_stride(primals_452, (160, ), (1, ))
    assert_size_stride(primals_453, (), ())
    assert_size_stride(primals_454, (160, ), (1, ))
    assert_size_stride(primals_455, (160, ), (1, ))
    assert_size_stride(primals_456, (), ())
    assert_size_stride(primals_457, (160, ), (1, ))
    assert_size_stride(primals_458, (160, ), (1, ))
    assert_size_stride(primals_459, (), ())
    assert_size_stride(primals_460, (192, ), (1, ))
    assert_size_stride(primals_461, (192, ), (1, ))
    assert_size_stride(primals_462, (), ())
    assert_size_stride(primals_463, (192, ), (1, ))
    assert_size_stride(primals_464, (192, ), (1, ))
    assert_size_stride(primals_465, (), ())
    assert_size_stride(primals_466, (192, ), (1, ))
    assert_size_stride(primals_467, (192, ), (1, ))
    assert_size_stride(primals_468, (), ())
    assert_size_stride(primals_469, (192, ), (1, ))
    assert_size_stride(primals_470, (192, ), (1, ))
    assert_size_stride(primals_471, (), ())
    assert_size_stride(primals_472, (192, ), (1, ))
    assert_size_stride(primals_473, (192, ), (1, ))
    assert_size_stride(primals_474, (), ())
    assert_size_stride(primals_475, (192, ), (1, ))
    assert_size_stride(primals_476, (192, ), (1, ))
    assert_size_stride(primals_477, (), ())
    assert_size_stride(primals_478, (192, ), (1, ))
    assert_size_stride(primals_479, (192, ), (1, ))
    assert_size_stride(primals_480, (), ())
    assert_size_stride(primals_481, (192, ), (1, ))
    assert_size_stride(primals_482, (192, ), (1, ))
    assert_size_stride(primals_483, (), ())
    assert_size_stride(primals_484, (192, ), (1, ))
    assert_size_stride(primals_485, (192, ), (1, ))
    assert_size_stride(primals_486, (), ())
    assert_size_stride(primals_487, (192, ), (1, ))
    assert_size_stride(primals_488, (192, ), (1, ))
    assert_size_stride(primals_489, (), ())
    assert_size_stride(primals_490, (192, ), (1, ))
    assert_size_stride(primals_491, (192, ), (1, ))
    assert_size_stride(primals_492, (), ())
    assert_size_stride(primals_493, (192, ), (1, ))
    assert_size_stride(primals_494, (192, ), (1, ))
    assert_size_stride(primals_495, (), ())
    assert_size_stride(primals_496, (192, ), (1, ))
    assert_size_stride(primals_497, (192, ), (1, ))
    assert_size_stride(primals_498, (), ())
    assert_size_stride(primals_499, (320, ), (1, ))
    assert_size_stride(primals_500, (320, ), (1, ))
    assert_size_stride(primals_501, (), ())
    assert_size_stride(primals_502, (192, ), (1, ))
    assert_size_stride(primals_503, (192, ), (1, ))
    assert_size_stride(primals_504, (), ())
    assert_size_stride(primals_505, (192, ), (1, ))
    assert_size_stride(primals_506, (192, ), (1, ))
    assert_size_stride(primals_507, (), ())
    assert_size_stride(primals_508, (192, ), (1, ))
    assert_size_stride(primals_509, (192, ), (1, ))
    assert_size_stride(primals_510, (), ())
    assert_size_stride(primals_511, (192, ), (1, ))
    assert_size_stride(primals_512, (192, ), (1, ))
    assert_size_stride(primals_513, (), ())
    assert_size_stride(primals_514, (320, ), (1, ))
    assert_size_stride(primals_515, (320, ), (1, ))
    assert_size_stride(primals_516, (), ())
    assert_size_stride(primals_517, (384, ), (1, ))
    assert_size_stride(primals_518, (384, ), (1, ))
    assert_size_stride(primals_519, (), ())
    assert_size_stride(primals_520, (384, ), (1, ))
    assert_size_stride(primals_521, (384, ), (1, ))
    assert_size_stride(primals_522, (), ())
    assert_size_stride(primals_523, (384, ), (1, ))
    assert_size_stride(primals_524, (384, ), (1, ))
    assert_size_stride(primals_525, (), ())
    assert_size_stride(primals_526, (448, ), (1, ))
    assert_size_stride(primals_527, (448, ), (1, ))
    assert_size_stride(primals_528, (), ())
    assert_size_stride(primals_529, (384, ), (1, ))
    assert_size_stride(primals_530, (384, ), (1, ))
    assert_size_stride(primals_531, (), ())
    assert_size_stride(primals_532, (384, ), (1, ))
    assert_size_stride(primals_533, (384, ), (1, ))
    assert_size_stride(primals_534, (), ())
    assert_size_stride(primals_535, (384, ), (1, ))
    assert_size_stride(primals_536, (384, ), (1, ))
    assert_size_stride(primals_537, (), ())
    assert_size_stride(primals_538, (192, ), (1, ))
    assert_size_stride(primals_539, (192, ), (1, ))
    assert_size_stride(primals_540, (), ())
    assert_size_stride(primals_541, (320, ), (1, ))
    assert_size_stride(primals_542, (320, ), (1, ))
    assert_size_stride(primals_543, (), ())
    assert_size_stride(primals_544, (384, ), (1, ))
    assert_size_stride(primals_545, (384, ), (1, ))
    assert_size_stride(primals_546, (), ())
    assert_size_stride(primals_547, (384, ), (1, ))
    assert_size_stride(primals_548, (384, ), (1, ))
    assert_size_stride(primals_549, (), ())
    assert_size_stride(primals_550, (384, ), (1, ))
    assert_size_stride(primals_551, (384, ), (1, ))
    assert_size_stride(primals_552, (), ())
    assert_size_stride(primals_553, (448, ), (1, ))
    assert_size_stride(primals_554, (448, ), (1, ))
    assert_size_stride(primals_555, (), ())
    assert_size_stride(primals_556, (384, ), (1, ))
    assert_size_stride(primals_557, (384, ), (1, ))
    assert_size_stride(primals_558, (), ())
    assert_size_stride(primals_559, (384, ), (1, ))
    assert_size_stride(primals_560, (384, ), (1, ))
    assert_size_stride(primals_561, (), ())
    assert_size_stride(primals_562, (384, ), (1, ))
    assert_size_stride(primals_563, (384, ), (1, ))
    assert_size_stride(primals_564, (), ())
    assert_size_stride(primals_565, (192, ), (1, ))
    assert_size_stride(primals_566, (192, ), (1, ))
    assert_size_stride(primals_567, (8, 3, 299, 299), (268203, 89401, 299, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_189, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_189
        buf1 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_190, buf1, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_190
        buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_191, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_191
        buf3 = empty_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_193, buf3, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del primals_193
        buf4 = empty_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_196, buf4, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_196
        buf5 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_198, buf5, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_198
        buf6 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_199, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_199
        buf7 = empty_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_203, buf7, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_203
        buf8 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_205, buf8, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_205
        buf9 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_206, buf9, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_206
        buf10 = empty_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_210, buf10, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_210
        buf11 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_212, buf11, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_212
        buf12 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_213, buf12, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_213
        buf13 = empty_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_215, buf13, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del primals_215
        buf14 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_217, buf14, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_217
        buf15 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_218, buf15, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_218
        buf16 = empty_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_221, buf16, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_221
        buf17 = empty_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_222, buf17, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_222
        buf18 = empty_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_224, buf18, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_224
        buf19 = empty_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_225, buf19, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_225
        buf20 = empty_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_226, buf20, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_226
        buf21 = empty_strided((192, 128, 1, 7), (896, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_227, buf21, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_227
        buf22 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_231, buf22, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_231
        buf23 = empty_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_232, buf23, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_232
        buf24 = empty_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_234, buf24, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_234
        buf25 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_235, buf25, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_235
        buf26 = empty_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_236, buf26, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_236
        buf27 = empty_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_237, buf27, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_237
        buf28 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_241, buf28, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_241
        buf29 = empty_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_242, buf29, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_242
        buf30 = empty_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_244, buf30, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_244
        buf31 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_245, buf31, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_245
        buf32 = empty_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_246, buf32, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_246
        buf33 = empty_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_247, buf33, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_247
        buf34 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_251, buf34, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_251
        buf35 = empty_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_252, buf35, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_252
        buf36 = empty_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_254, buf36, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_254
        buf37 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_255, buf37, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_255
        buf38 = empty_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_256, buf38, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_256
        buf39 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_257, buf39, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_257
        buf40 = empty_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_13.run(primals_260, buf40, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del primals_260
        buf41 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_262, buf41, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_262
        buf42 = empty_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_263, buf42, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_263
        buf43 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(primals_264, buf43, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_264
        buf44 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_267, buf44, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_267
        buf45 = empty_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_268, buf45, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_268
        buf46 = empty_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_270, buf46, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_270
        buf47 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_271, buf47, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_271
        buf48 = empty_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_272, buf48, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_272
        buf49 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_276, buf49, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_276
        buf50 = empty_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_277, buf50, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_277
        buf51 = empty_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_279, buf51, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_279
        buf52 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_280, buf52, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_280
        buf53 = empty_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_281, buf53, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_281
        buf54 = empty_strided((8, 3, 299, 299), (268203, 1, 897, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(primals_567, buf54, 24, 89401, grid=grid(24, 89401), stream=stream0)
        del primals_567
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 32, 149, 149), (710432, 22201, 149, 1))
        buf56 = empty_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf55, buf56, 256, 22201, grid=grid(256, 22201), stream=stream0)
        buf57 = empty_strided((1, 32, 1, 1, 1388), (44416, 1, 44416, 44416, 32), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 32, 1, 1, 1388), (44416, 1, 44416, 44416, 32), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((1, 32, 1, 1, 1388), (44416, 1, 44416, 44416, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf56, buf57, buf58, buf59, 44416, 128, grid=grid(44416), stream=stream0)
        buf60 = empty_strided((1, 32, 1, 1, 11), (352, 1, 352, 352, 32), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 32, 1, 1, 11), (352, 1, 352, 352, 32), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 32, 1, 1, 11), (352, 1, 352, 352, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf57, buf58, buf59, buf60, buf61, buf62, 352, 127, grid=grid(352), stream=stream0)
        del buf57
        del buf58
        del buf59
        buf63 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf64 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf66 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf60, buf61, buf62, primals_286, primals_287, buf63, buf64, buf66, primals_286, primals_287, 32, 11, grid=grid(32), stream=stream0)
        del primals_286
        del primals_287
        buf67 = reinterpret_tensor(buf55, (8, 32, 149, 149), (710432, 1, 4768, 32), 0); del buf55  # reuse
        # Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_22.run(buf56, buf63, buf64, primals_1, primals_2, buf67, 5683456, grid=grid(5683456), stream=stream0)
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 32, 147, 147), (691488, 21609, 147, 1))
        buf69 = empty_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf68, buf69, 256, 21609, grid=grid(256, 21609), stream=stream0)
        buf70 = empty_strided((1, 32, 1, 1, 1351), (43232, 1, 43232, 43232, 32), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 32, 1, 1, 1351), (43232, 1, 43232, 43232, 32), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 32, 1, 1, 1351), (43232, 1, 43232, 43232, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf69, buf70, buf71, buf72, 43232, 128, grid=grid(43232), stream=stream0)
        buf73 = buf62; del buf62  # reuse
        buf74 = buf61; del buf61  # reuse
        buf75 = buf60; del buf60  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf70, buf71, buf72, buf73, buf74, buf75, 352, 123, grid=grid(352), stream=stream0)
        del buf70
        del buf71
        del buf72
        buf76 = buf64; del buf64  # reuse
        buf77 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf79 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf73, buf74, buf75, primals_289, primals_290, buf76, buf77, buf79, primals_289, primals_290, 32, 11, grid=grid(32), stream=stream0)
        del buf73
        del buf74
        del buf75
        del primals_289
        del primals_290
        buf80 = reinterpret_tensor(buf68, (8, 32, 147, 147), (691488, 1, 4704, 32), 0); del buf68  # reuse
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf69, buf76, buf77, primals_3, primals_4, buf80, 5531904, grid=grid(5531904), stream=stream0)
        del primals_4
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 64, 147, 147), (1382976, 21609, 147, 1))
        buf82 = empty_strided((8, 64, 147, 147), (1382976, 1, 9408, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf81, buf82, 512, 21609, grid=grid(512, 21609), stream=stream0)
        buf83 = empty_strided((1, 64, 1, 1, 1029), (65856, 1, 65856, 65856, 64), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((1, 64, 1, 1, 1029), (65856, 1, 65856, 65856, 64), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((1, 64, 1, 1, 1029), (65856, 1, 65856, 65856, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf82, buf83, buf84, buf85, 65856, 168, grid=grid(65856), stream=stream0)
        buf86 = empty_strided((1, 64, 1, 1, 9), (576, 1, 576, 576, 64), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((1, 64, 1, 1, 9), (576, 1, 576, 576, 64), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((1, 64, 1, 1, 9), (576, 1, 576, 576, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf83, buf84, buf85, buf86, buf87, buf88, 576, 115, grid=grid(576), stream=stream0)
        del buf83
        del buf84
        del buf85
        buf89 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf92 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf86, buf87, buf88, primals_292, primals_293, buf89, buf90, buf92, primals_292, primals_293, 64, 9, grid=grid(64), stream=stream0)
        del primals_292
        del primals_293
        buf93 = reinterpret_tensor(buf81, (8, 64, 147, 147), (1382976, 1, 9408, 64), 0); del buf81  # reuse
        # Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf82, buf89, buf90, primals_5, primals_6, buf93, 11063808, grid=grid(11063808), stream=stream0)
        del primals_6
        buf94 = empty_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cuda', dtype=torch.float32)
        buf95 = empty_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_18], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_33.run(buf93, buf94, buf95, 2728448, grid=grid(2728448), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf94, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 80, 73, 73), (426320, 5329, 73, 1))
        buf97 = empty_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf96, buf97, 640, 5329, grid=grid(640, 5329), stream=stream0)
        buf98 = empty_strided((1, 80, 1, 1, 334), (26720, 1, 26720, 26720, 80), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((1, 80, 1, 1, 334), (26720, 1, 26720, 26720, 80), device='cuda', dtype=torch.float32)
        buf100 = empty_strided((1, 80, 1, 1, 334), (26720, 1, 26720, 26720, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf97, buf98, buf99, buf100, 26720, 128, grid=grid(26720), stream=stream0)
        buf101 = empty_strided((1, 80, 1, 1, 3), (240, 1, 240, 240, 80), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((1, 80, 1, 1, 3), (240, 1, 240, 240, 80), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((1, 80, 1, 1, 3), (240, 1, 240, 240, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf98, buf99, buf100, buf101, buf102, buf103, 240, 112, grid=grid(240), stream=stream0)
        del buf100
        del buf98
        del buf99
        buf104 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf107 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf101, buf102, buf103, primals_295, primals_296, buf104, buf105, buf107, primals_295, primals_296, 80, 3, grid=grid(80), stream=stream0)
        del buf101
        del buf102
        del buf103
        del primals_295
        del primals_296
        buf108 = reinterpret_tensor(buf96, (8, 80, 73, 73), (426320, 1, 5840, 80), 0); del buf96  # reuse
        # Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf97, buf104, buf105, primals_7, primals_8, buf108, 3410560, grid=grid(3410560), stream=stream0)
        del buf105
        del primals_8
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, buf3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 192, 71, 71), (967872, 5041, 71, 1))
        buf110 = empty_strided((8, 192, 71, 71), (967872, 1, 13632, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf109, buf110, 1536, 5041, grid=grid(1536, 5041), stream=stream0)
        buf111 = empty_strided((1, 192, 1, 1, 316), (60672, 1, 60672, 60672, 192), device='cuda', dtype=torch.float32)
        buf112 = empty_strided((1, 192, 1, 1, 316), (60672, 1, 60672, 60672, 192), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((1, 192, 1, 1, 316), (60672, 1, 60672, 60672, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf110, buf111, buf112, buf113, 60672, 128, grid=grid(60672), stream=stream0)
        buf114 = reinterpret_tensor(buf88, (1, 192, 1, 1, 3), (576, 1, 576, 576, 192), 0); del buf88  # reuse
        buf115 = reinterpret_tensor(buf87, (1, 192, 1, 1, 3), (576, 1, 576, 576, 192), 0); del buf87  # reuse
        buf116 = reinterpret_tensor(buf86, (1, 192, 1, 1, 3), (576, 1, 576, 576, 192), 0); del buf86  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf111, buf112, buf113, buf114, buf115, buf116, 576, 106, grid=grid(576), stream=stream0)
        del buf111
        del buf112
        del buf113
        buf117 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf120 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf114, buf115, buf116, primals_298, primals_299, buf117, buf118, buf120, primals_298, primals_299, 192, 3, grid=grid(192), stream=stream0)
        del buf114
        del buf115
        del buf116
        del primals_298
        del primals_299
        buf121 = reinterpret_tensor(buf109, (8, 192, 71, 71), (967872, 1, 13632, 192), 0); del buf109  # reuse
        # Source Nodes: [x_26, x_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf110, buf117, buf118, primals_9, primals_10, buf121, 7742976, grid=grid(7742976), stream=stream0)
        del primals_10
        buf122 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_31], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_44.run(buf121, buf122, buf123, 1881600, grid=grid(1881600), stream=stream0)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf122, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf125 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf124, buf125, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf126 = empty_strided((1, 64, 1, 1, 77), (4928, 1, 4928, 4928, 64), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((1, 64, 1, 1, 77), (4928, 1, 4928, 4928, 64), device='cuda', dtype=torch.float32)
        buf128 = empty_strided((1, 64, 1, 1, 77), (4928, 1, 4928, 4928, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf125, buf126, buf127, buf128, 4928, 128, grid=grid(4928), stream=stream0)
        buf129 = buf90; del buf90  # reuse
        buf130 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf132 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf126, buf127, buf128, primals_301, primals_302, buf129, buf130, buf132, primals_301, primals_302, 64, 77, grid=grid(64), stream=stream0)
        del primals_301
        del primals_302
        buf195 = empty((8, 256, 35, 35), device='cuda', dtype=torch.float32)
        buf133 = reinterpret_tensor(buf195, (8, 64, 35, 35), (313600, 1225, 35, 1), 0)  # alias
        buf1102 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48.run(buf125, buf129, buf130, primals_11, primals_12, buf133, buf1102, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_12
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf122, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 48, 35, 35), (58800, 1225, 35, 1))
        buf135 = empty_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf134, buf135, 384, 1225, grid=grid(384, 1225), stream=stream0)
        buf136 = empty_strided((1, 48, 1, 1, 77), (3696, 1, 3696, 3696, 48), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((1, 48, 1, 1, 77), (3696, 1, 3696, 3696, 48), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1, 48, 1, 1, 77), (3696, 1, 3696, 3696, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf135, buf136, buf137, buf138, 3696, 128, grid=grid(3696), stream=stream0)
        buf139 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf140 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf142 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf136, buf137, buf138, primals_304, primals_305, buf139, buf140, buf142, primals_304, primals_305, 48, 77, grid=grid(48), stream=stream0)
        del primals_304
        del primals_305
        buf143 = reinterpret_tensor(buf134, (8, 48, 35, 35), (58800, 1, 1680, 48), 0); del buf134  # reuse
        # Source Nodes: [branch5x5, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf135, buf139, buf140, primals_13, primals_14, buf143, 470400, grid=grid(470400), stream=stream0)
        del primals_14
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, buf4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf145 = reinterpret_tensor(buf124, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf124  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf144, buf145, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf146 = buf128; del buf128  # reuse
        buf147 = buf127; del buf127  # reuse
        buf148 = buf126; del buf126  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf145, buf146, buf147, buf148, 4928, 128, grid=grid(4928), stream=stream0)
        buf149 = buf130; del buf130  # reuse
        buf150 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf152 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf146, buf147, buf148, primals_307, primals_308, buf149, buf150, buf152, primals_307, primals_308, 64, 77, grid=grid(64), stream=stream0)
        del primals_307
        del primals_308
        buf153 = reinterpret_tensor(buf195, (8, 64, 35, 35), (313600, 1225, 35, 1), 78400)  # alias
        buf1101 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch5x5_1, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48.run(buf145, buf149, buf150, primals_15, primals_16, buf153, buf1101, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_16
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf122, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf155 = reinterpret_tensor(buf144, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf144  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf154, buf155, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf156 = buf148; del buf148  # reuse
        buf157 = buf147; del buf147  # reuse
        buf158 = buf146; del buf146  # reuse
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf155, buf156, buf157, buf158, 4928, 128, grid=grid(4928), stream=stream0)
        buf159 = buf150; del buf150  # reuse
        buf160 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf162 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf156, buf157, buf158, primals_310, primals_311, buf159, buf160, buf162, primals_310, primals_311, 64, 77, grid=grid(64), stream=stream0)
        del primals_310
        del primals_311
        buf163 = reinterpret_tensor(buf154, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf154  # reuse
        # Source Nodes: [branch3x3dbl, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf155, buf159, buf160, primals_17, primals_18, buf163, 627200, grid=grid(627200), stream=stream0)
        del primals_18
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf165 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf164, buf165, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf166 = empty_strided((1, 96, 1, 1, 77), (7392, 1, 7392, 7392, 96), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 96, 1, 1, 77), (7392, 1, 7392, 7392, 96), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((1, 96, 1, 1, 77), (7392, 1, 7392, 7392, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf165, buf166, buf167, buf168, 7392, 128, grid=grid(7392), stream=stream0)
        buf169 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf172 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf166, buf167, buf168, primals_313, primals_314, buf169, buf170, buf172, primals_313, primals_314, 96, 77, grid=grid(96), stream=stream0)
        del primals_313
        del primals_314
        buf173 = reinterpret_tensor(buf164, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf164  # reuse
        # Source Nodes: [branch3x3dbl_1, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf165, buf169, buf170, primals_19, primals_20, buf173, 940800, grid=grid(940800), stream=stream0)
        del primals_20
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf175 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf174, buf175, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf176 = buf168; del buf168  # reuse
        buf177 = buf167; del buf167  # reuse
        buf178 = buf166; del buf166  # reuse
        # Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf175, buf176, buf177, buf178, 7392, 128, grid=grid(7392), stream=stream0)
        buf179 = buf170; del buf170  # reuse
        buf180 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf182 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf176, buf177, buf178, primals_316, primals_317, buf179, buf180, buf182, primals_316, primals_317, 96, 77, grid=grid(96), stream=stream0)
        del primals_316
        del primals_317
        buf183 = reinterpret_tensor(buf195, (8, 96, 35, 35), (313600, 1225, 35, 1), 156800)  # alias
        buf1100 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3dbl_2, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_58.run(buf175, buf179, buf180, primals_21, primals_22, buf183, buf1100, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del primals_22
        buf184 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_59.run(buf122, buf184, 1536, 1225, grid=grid(1536, 1225), stream=stream0)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 32, 35, 35), (39200, 1225, 35, 1))
        buf186 = empty_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf185, buf186, 256, 1225, grid=grid(256, 1225), stream=stream0)
        del buf185
        buf187 = empty_strided((1, 32, 1, 1, 77), (2464, 1, 2464, 2464, 32), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((1, 32, 1, 1, 77), (2464, 1, 2464, 2464, 32), device='cuda', dtype=torch.float32)
        buf189 = empty_strided((1, 32, 1, 1, 77), (2464, 1, 2464, 2464, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf186, buf187, buf188, buf189, 2464, 128, grid=grid(2464), stream=stream0)
        buf190 = buf77; del buf77  # reuse
        buf191 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf193 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf187, buf188, buf189, primals_319, primals_320, buf190, buf191, buf193, primals_319, primals_320, 32, 77, grid=grid(32), stream=stream0)
        del buf187
        del buf188
        del buf189
        del primals_319
        del primals_320
        buf194 = reinterpret_tensor(buf195, (8, 32, 35, 35), (313600, 1225, 35, 1), 274400)  # alias
        buf1099 = empty_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_1, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_63.run(buf186, buf190, buf191, primals_23, primals_24, buf194, buf1099, 256, 1225, grid=grid(256, 1225), stream=stream0)
        del buf191
        del primals_24
        buf196 = empty_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf195, buf196, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del buf133
        del buf153
        del buf183
        del buf194
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf198 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf197, buf198, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf199 = buf158; del buf158  # reuse
        buf200 = buf157; del buf157  # reuse
        buf201 = buf156; del buf156  # reuse
        # Source Nodes: [x_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf198, buf199, buf200, buf201, 4928, 128, grid=grid(4928), stream=stream0)
        buf202 = buf160; del buf160  # reuse
        buf203 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf205 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf199, buf200, buf201, primals_322, primals_323, buf202, buf203, buf205, primals_322, primals_323, 64, 77, grid=grid(64), stream=stream0)
        del primals_322
        del primals_323
        buf268 = empty((8, 288, 35, 35), device='cuda', dtype=torch.float32)
        buf206 = reinterpret_tensor(buf268, (8, 64, 35, 35), (352800, 1225, 35, 1), 0)  # alias
        buf1098 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_1, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf198, buf202, buf203, primals_25, primals_26, buf206, buf1098, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_26
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf196, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 48, 35, 35), (58800, 1225, 35, 1))
        buf208 = empty_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf207, buf208, 384, 1225, grid=grid(384, 1225), stream=stream0)
        buf209 = buf138; del buf138  # reuse
        buf210 = buf137; del buf137  # reuse
        buf211 = buf136; del buf136  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf208, buf209, buf210, buf211, 3696, 128, grid=grid(3696), stream=stream0)
        buf212 = buf140; del buf140  # reuse
        buf213 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf215 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf209, buf210, buf211, primals_325, primals_326, buf212, buf213, buf215, primals_325, primals_326, 48, 77, grid=grid(48), stream=stream0)
        del primals_325
        del primals_326
        buf216 = reinterpret_tensor(buf207, (8, 48, 35, 35), (58800, 1, 1680, 48), 0); del buf207  # reuse
        # Source Nodes: [branch5x5_2, x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf208, buf212, buf213, primals_27, primals_28, buf216, 470400, grid=grid(470400), stream=stream0)
        del primals_28
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, buf7, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf218 = reinterpret_tensor(buf197, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf197  # reuse
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf217, buf218, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf219 = buf201; del buf201  # reuse
        buf220 = buf200; del buf200  # reuse
        buf221 = buf199; del buf199  # reuse
        # Source Nodes: [x_79], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf218, buf219, buf220, buf221, 4928, 128, grid=grid(4928), stream=stream0)
        buf222 = buf203; del buf203  # reuse
        buf223 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf225 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf219, buf220, buf221, primals_328, primals_329, buf222, buf223, buf225, primals_328, primals_329, 64, 77, grid=grid(64), stream=stream0)
        del primals_328
        del primals_329
        buf226 = reinterpret_tensor(buf268, (8, 64, 35, 35), (352800, 1225, 35, 1), 78400)  # alias
        buf1097 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch5x5_3, x_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf218, buf222, buf223, primals_29, primals_30, buf226, buf1097, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_30
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf196, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf228 = reinterpret_tensor(buf217, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf217  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf227, buf228, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf229 = buf221; del buf221  # reuse
        buf230 = buf220; del buf220  # reuse
        buf231 = buf219; del buf219  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf228, buf229, buf230, buf231, 4928, 128, grid=grid(4928), stream=stream0)
        buf232 = buf223; del buf223  # reuse
        buf233 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf235 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf229, buf230, buf231, primals_331, primals_332, buf232, buf233, buf235, primals_331, primals_332, 64, 77, grid=grid(64), stream=stream0)
        del primals_331
        del primals_332
        buf236 = reinterpret_tensor(buf227, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf227  # reuse
        # Source Nodes: [branch3x3dbl_3, x_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf228, buf232, buf233, primals_31, primals_32, buf236, 627200, grid=grid(627200), stream=stream0)
        del primals_32
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf238 = reinterpret_tensor(buf174, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf174  # reuse
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf237, buf238, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf239 = buf178; del buf178  # reuse
        buf240 = buf177; del buf177  # reuse
        buf241 = buf176; del buf176  # reuse
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf238, buf239, buf240, buf241, 7392, 128, grid=grid(7392), stream=stream0)
        buf242 = buf180; del buf180  # reuse
        buf243 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf245 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf239, buf240, buf241, primals_334, primals_335, buf242, buf243, buf245, primals_334, primals_335, 96, 77, grid=grid(96), stream=stream0)
        del primals_334
        del primals_335
        buf246 = reinterpret_tensor(buf237, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf237  # reuse
        # Source Nodes: [branch3x3dbl_4, x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf238, buf242, buf243, primals_33, primals_34, buf246, 940800, grid=grid(940800), stream=stream0)
        del primals_34
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf248 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf247, buf248, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf249 = buf241; del buf241  # reuse
        buf250 = buf240; del buf240  # reuse
        buf251 = buf239; del buf239  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf248, buf249, buf250, buf251, 7392, 128, grid=grid(7392), stream=stream0)
        buf252 = buf243; del buf243  # reuse
        buf253 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf255 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf249, buf250, buf251, primals_337, primals_338, buf252, buf253, buf255, primals_337, primals_338, 96, 77, grid=grid(96), stream=stream0)
        del primals_337
        del primals_338
        buf256 = reinterpret_tensor(buf268, (8, 96, 35, 35), (352800, 1225, 35, 1), 156800)  # alias
        buf1096 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3dbl_5, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66.run(buf248, buf252, buf253, primals_35, primals_36, buf256, buf1096, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del primals_36
        buf257 = reinterpret_tensor(buf195, (8, 256, 35, 35), (313600, 1, 8960, 256), 0); del buf195  # reuse
        # Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_67.run(buf196, buf257, 2508800, grid=grid(2508800), stream=stream0)
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf259 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf258, buf259, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf260 = buf231; del buf231  # reuse
        buf261 = buf230; del buf230  # reuse
        buf262 = buf229; del buf229  # reuse
        # Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf259, buf260, buf261, buf262, 4928, 128, grid=grid(4928), stream=stream0)
        buf263 = buf233; del buf233  # reuse
        buf264 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf266 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf260, buf261, buf262, primals_340, primals_341, buf263, buf264, buf266, primals_340, primals_341, 64, 77, grid=grid(64), stream=stream0)
        del primals_340
        del primals_341
        buf267 = reinterpret_tensor(buf268, (8, 64, 35, 35), (352800, 1225, 35, 1), 274400)  # alias
        buf1095 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_3, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf259, buf263, buf264, primals_37, primals_38, buf267, buf1095, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_38
        buf269 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf268, buf269, 2304, 1225, grid=grid(2304, 1225), stream=stream0)
        del buf206
        del buf226
        del buf256
        del buf267
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf271 = reinterpret_tensor(buf258, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf258  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf270, buf271, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf272 = buf262; del buf262  # reuse
        buf273 = buf261; del buf261  # reuse
        buf274 = buf260; del buf260  # reuse
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf271, buf272, buf273, buf274, 4928, 128, grid=grid(4928), stream=stream0)
        buf275 = buf264; del buf264  # reuse
        buf276 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf278 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf272, buf273, buf274, primals_343, primals_344, buf275, buf276, buf278, primals_343, primals_344, 64, 77, grid=grid(64), stream=stream0)
        del primals_343
        del primals_344
        buf341 = buf268; del buf268  # reuse
        buf279 = reinterpret_tensor(buf341, (8, 64, 35, 35), (352800, 1225, 35, 1), 0)  # alias
        buf1094 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_2, x_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf271, buf275, buf276, primals_39, primals_40, buf279, buf1094, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_40
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf269, primals_209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 48, 35, 35), (58800, 1225, 35, 1))
        buf281 = empty_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf280, buf281, 384, 1225, grid=grid(384, 1225), stream=stream0)
        buf282 = buf211; del buf211  # reuse
        buf283 = buf210; del buf210  # reuse
        buf284 = buf209; del buf209  # reuse
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf281, buf282, buf283, buf284, 3696, 128, grid=grid(3696), stream=stream0)
        buf285 = buf213; del buf213  # reuse
        buf286 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf288 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf282, buf283, buf284, primals_346, primals_347, buf285, buf286, buf288, primals_346, primals_347, 48, 77, grid=grid(48), stream=stream0)
        del buf282
        del buf283
        del buf284
        del primals_346
        del primals_347
        buf289 = reinterpret_tensor(buf280, (8, 48, 35, 35), (58800, 1, 1680, 48), 0); del buf280  # reuse
        # Source Nodes: [branch5x5_4, x_110], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf281, buf285, buf286, primals_41, primals_42, buf289, 470400, grid=grid(470400), stream=stream0)
        del buf286
        del primals_42
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, buf10, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf291 = reinterpret_tensor(buf270, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf270  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf290, buf291, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf292 = buf274; del buf274  # reuse
        buf293 = buf273; del buf273  # reuse
        buf294 = buf272; del buf272  # reuse
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf291, buf292, buf293, buf294, 4928, 128, grid=grid(4928), stream=stream0)
        buf295 = buf276; del buf276  # reuse
        buf296 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf298 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf292, buf293, buf294, primals_349, primals_350, buf295, buf296, buf298, primals_349, primals_350, 64, 77, grid=grid(64), stream=stream0)
        del primals_349
        del primals_350
        buf299 = reinterpret_tensor(buf341, (8, 64, 35, 35), (352800, 1225, 35, 1), 78400)  # alias
        buf1093 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch5x5_5, x_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf291, buf295, buf296, primals_43, primals_44, buf299, buf1093, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_44
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf269, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf301 = reinterpret_tensor(buf290, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf290  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf300, buf301, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf302 = buf294; del buf294  # reuse
        buf303 = buf293; del buf293  # reuse
        buf304 = buf292; del buf292  # reuse
        # Source Nodes: [x_120], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf301, buf302, buf303, buf304, 4928, 128, grid=grid(4928), stream=stream0)
        buf305 = buf296; del buf296  # reuse
        buf306 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf308 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf302, buf303, buf304, primals_352, primals_353, buf305, buf306, buf308, primals_352, primals_353, 64, 77, grid=grid(64), stream=stream0)
        del primals_352
        del primals_353
        buf309 = reinterpret_tensor(buf300, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf300  # reuse
        # Source Nodes: [branch3x3dbl_6, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf301, buf305, buf306, primals_45, primals_46, buf309, 627200, grid=grid(627200), stream=stream0)
        del primals_46
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf311 = reinterpret_tensor(buf247, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf247  # reuse
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf310, buf311, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf312 = buf251; del buf251  # reuse
        buf313 = buf250; del buf250  # reuse
        buf314 = buf249; del buf249  # reuse
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf311, buf312, buf313, buf314, 7392, 128, grid=grid(7392), stream=stream0)
        buf315 = buf253; del buf253  # reuse
        buf316 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf318 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf312, buf313, buf314, primals_355, primals_356, buf315, buf316, buf318, primals_355, primals_356, 96, 77, grid=grid(96), stream=stream0)
        del primals_355
        del primals_356
        buf319 = reinterpret_tensor(buf310, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf310  # reuse
        # Source Nodes: [branch3x3dbl_7, x_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf311, buf315, buf316, primals_47, primals_48, buf319, 940800, grid=grid(940800), stream=stream0)
        del primals_48
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf321 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf320, buf321, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf322 = buf314; del buf314  # reuse
        buf323 = buf313; del buf313  # reuse
        buf324 = buf312; del buf312  # reuse
        # Source Nodes: [x_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf321, buf322, buf323, buf324, 7392, 128, grid=grid(7392), stream=stream0)
        buf325 = buf316; del buf316  # reuse
        buf326 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf328 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf322, buf323, buf324, primals_358, primals_359, buf325, buf326, buf328, primals_358, primals_359, 96, 77, grid=grid(96), stream=stream0)
        del primals_358
        del primals_359
        buf329 = reinterpret_tensor(buf341, (8, 96, 35, 35), (352800, 1225, 35, 1), 156800)  # alias
        buf1092 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3dbl_8, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66.run(buf321, buf325, buf326, primals_49, primals_50, buf329, buf1092, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del primals_50
        buf330 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_69.run(buf269, buf330, 2822400, grid=grid(2822400), stream=stream0)
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf332 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf331, buf332, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf333 = buf304; del buf304  # reuse
        buf334 = buf303; del buf303  # reuse
        buf335 = buf302; del buf302  # reuse
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf332, buf333, buf334, buf335, 4928, 128, grid=grid(4928), stream=stream0)
        buf336 = buf306; del buf306  # reuse
        buf337 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf339 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf333, buf334, buf335, primals_361, primals_362, buf336, buf337, buf339, primals_361, primals_362, 64, 77, grid=grid(64), stream=stream0)
        del primals_361
        del primals_362
        buf340 = reinterpret_tensor(buf341, (8, 64, 35, 35), (352800, 1225, 35, 1), 274400)  # alias
        buf1091 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_5, x_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf332, buf336, buf337, primals_51, primals_52, buf340, buf1091, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del primals_52
        buf342 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf341, buf342, 2304, 1225, grid=grid(2304, 1225), stream=stream0)
        del buf279
        del buf299
        del buf329
        del buf340
        del buf341
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, buf13, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (8, 384, 17, 17), (110976, 289, 17, 1))
        buf344 = empty_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf343, buf344, 3072, 289, grid=grid(3072, 289), stream=stream0)
        del buf343
        buf345 = empty_strided((1, 384, 1, 1, 19), (7296, 1, 7296, 7296, 384), device='cuda', dtype=torch.float32)
        buf346 = empty_strided((1, 384, 1, 1, 19), (7296, 1, 7296, 7296, 384), device='cuda', dtype=torch.float32)
        buf347 = empty_strided((1, 384, 1, 1, 19), (7296, 1, 7296, 7296, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf344, buf345, buf346, buf347, 7296, 122, grid=grid(7296), stream=stream0)
        buf348 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf351 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_72.run(buf345, buf346, buf347, primals_364, primals_365, buf348, buf349, buf351, primals_364, primals_365, 384, 19, grid=grid(384), stream=stream0)
        del buf345
        del buf346
        del buf347
        del primals_364
        del primals_365
        buf385 = empty((8, 768, 17, 17), device='cuda', dtype=torch.float32)
        buf352 = reinterpret_tensor(buf385, (8, 384, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1090 = empty_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3, x_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73.run(buf344, buf348, buf349, primals_53, primals_54, buf352, buf1090, 3072, 289, grid=grid(3072, 289), stream=stream0)
        del primals_54
        # Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf342, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 64, 35, 35), (78400, 1225, 35, 1))
        buf354 = reinterpret_tensor(buf331, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf331  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf353, buf354, 512, 1225, grid=grid(512, 1225), stream=stream0)
        buf355 = buf335; del buf335  # reuse
        buf356 = buf334; del buf334  # reuse
        buf357 = buf333; del buf333  # reuse
        # Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf354, buf355, buf356, buf357, 4928, 128, grid=grid(4928), stream=stream0)
        buf358 = buf337; del buf337  # reuse
        buf359 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf361 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf355, buf356, buf357, primals_367, primals_368, buf358, buf359, buf361, primals_367, primals_368, 64, 77, grid=grid(64), stream=stream0)
        del buf355
        del buf356
        del buf357
        del primals_367
        del primals_368
        buf362 = reinterpret_tensor(buf353, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf353  # reuse
        # Source Nodes: [branch3x3dbl_9, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf354, buf358, buf359, primals_55, primals_56, buf362, 627200, grid=grid(627200), stream=stream0)
        del buf359
        del primals_56
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 96, 35, 35), (117600, 1225, 35, 1))
        buf364 = reinterpret_tensor(buf320, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf320  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf363, buf364, 768, 1225, grid=grid(768, 1225), stream=stream0)
        buf365 = buf324; del buf324  # reuse
        buf366 = buf323; del buf323  # reuse
        buf367 = buf322; del buf322  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf364, buf365, buf366, buf367, 7392, 128, grid=grid(7392), stream=stream0)
        buf368 = buf326; del buf326  # reuse
        buf369 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf371 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf365, buf366, buf367, primals_370, primals_371, buf368, buf369, buf371, primals_370, primals_371, 96, 77, grid=grid(96), stream=stream0)
        del buf365
        del buf366
        del buf367
        del primals_370
        del primals_371
        buf372 = reinterpret_tensor(buf363, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf363  # reuse
        # Source Nodes: [branch3x3dbl_10, x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf364, buf368, buf369, primals_57, primals_58, buf372, 940800, grid=grid(940800), stream=stream0)
        del primals_58
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, buf15, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 96, 17, 17), (27744, 289, 17, 1))
        buf374 = empty_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf373, buf374, 768, 289, grid=grid(768, 289), stream=stream0)
        del buf373
        buf375 = empty_strided((1, 96, 1, 1, 19), (1824, 1, 1824, 1824, 96), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((1, 96, 1, 1, 19), (1824, 1, 1824, 1824, 96), device='cuda', dtype=torch.float32)
        buf377 = empty_strided((1, 96, 1, 1, 19), (1824, 1, 1824, 1824, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf374, buf375, buf376, buf377, 1824, 122, grid=grid(1824), stream=stream0)
        buf378 = buf369; del buf369  # reuse
        buf379 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf381 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf375, buf376, buf377, primals_373, primals_374, buf378, buf379, buf381, primals_373, primals_374, 96, 19, grid=grid(96), stream=stream0)
        del buf375
        del buf376
        del buf377
        del primals_373
        del primals_374
        buf382 = reinterpret_tensor(buf385, (8, 96, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1089 = empty_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3dbl_11, x_156], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf374, buf378, buf379, primals_59, primals_60, buf382, buf1089, 768, 289, grid=grid(768, 289), stream=stream0)
        del buf379
        del primals_60
        buf383 = reinterpret_tensor(buf385, (8, 288, 17, 17), (221952, 289, 17, 1), 138720)  # alias
        # Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_78.run(buf342, buf383, 2304, 289, grid=grid(2304, 289), stream=stream0)
        buf384 = empty_strided((8, 288, 17, 17), (83232, 1, 4896, 288), device='cuda', dtype=torch.int64)
        # Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_79.run(buf342, buf384, 665856, grid=grid(665856), stream=stream0)
        buf386 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf385, buf386, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf352
        del buf382
        del buf383
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf388 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf387, buf388, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf389 = empty_strided((1, 192, 1, 1, 19), (3648, 1, 3648, 3648, 192), device='cuda', dtype=torch.float32)
        buf390 = empty_strided((1, 192, 1, 1, 19), (3648, 1, 3648, 3648, 192), device='cuda', dtype=torch.float32)
        buf391 = empty_strided((1, 192, 1, 1, 19), (3648, 1, 3648, 3648, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf388, buf389, buf390, buf391, 3648, 122, grid=grid(3648), stream=stream0)
        buf392 = buf118; del buf118  # reuse
        buf393 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf395 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf389, buf390, buf391, primals_376, primals_377, buf392, buf393, buf395, primals_376, primals_377, 192, 19, grid=grid(192), stream=stream0)
        del primals_376
        del primals_377
        buf488 = buf385; del buf385  # reuse
        buf396 = reinterpret_tensor(buf488, (8, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1088 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_3, x_162], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf388, buf392, buf393, primals_61, primals_62, buf396, buf1088, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_62
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf386, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf398 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf397, buf398, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf399 = empty_strided((1, 128, 1, 1, 19), (2432, 1, 2432, 2432, 128), device='cuda', dtype=torch.float32)
        buf400 = empty_strided((1, 128, 1, 1, 19), (2432, 1, 2432, 2432, 128), device='cuda', dtype=torch.float32)
        buf401 = empty_strided((1, 128, 1, 1, 19), (2432, 1, 2432, 2432, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf398, buf399, buf400, buf401, 2432, 122, grid=grid(2432), stream=stream0)
        buf402 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf403 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf405 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf399, buf400, buf401, primals_379, primals_380, buf402, buf403, buf405, primals_379, primals_380, 128, 19, grid=grid(128), stream=stream0)
        del primals_379
        del primals_380
        buf406 = reinterpret_tensor(buf397, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf397  # reuse
        # Source Nodes: [branch7x7, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf398, buf402, buf403, primals_63, primals_64, buf406, 295936, grid=grid(295936), stream=stream0)
        del primals_64
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, buf16, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf408 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf407, buf408, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf409 = buf401; del buf401  # reuse
        buf410 = buf400; del buf400  # reuse
        buf411 = buf399; del buf399  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf408, buf409, buf410, buf411, 2432, 122, grid=grid(2432), stream=stream0)
        buf412 = buf403; del buf403  # reuse
        buf413 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf415 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf409, buf410, buf411, primals_382, primals_383, buf412, buf413, buf415, primals_382, primals_383, 128, 19, grid=grid(128), stream=stream0)
        del primals_382
        del primals_383
        buf416 = reinterpret_tensor(buf407, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf407  # reuse
        # Source Nodes: [branch7x7_1, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf408, buf412, buf413, primals_65, primals_66, buf416, 295936, grid=grid(295936), stream=stream0)
        del primals_66
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, buf17, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf418 = reinterpret_tensor(buf387, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf387  # reuse
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf417, buf418, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf419 = buf391; del buf391  # reuse
        buf420 = buf390; del buf390  # reuse
        buf421 = buf389; del buf389  # reuse
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf418, buf419, buf420, buf421, 3648, 122, grid=grid(3648), stream=stream0)
        buf422 = buf393; del buf393  # reuse
        buf423 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf425 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf419, buf420, buf421, primals_385, primals_386, buf422, buf423, buf425, primals_385, primals_386, 192, 19, grid=grid(192), stream=stream0)
        del primals_385
        del primals_386
        buf426 = reinterpret_tensor(buf488, (8, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1087 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7_2, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf418, buf422, buf423, primals_67, primals_68, buf426, buf1087, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_68
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf386, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf428 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf427, buf428, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf429 = buf411; del buf411  # reuse
        buf430 = buf410; del buf410  # reuse
        buf431 = buf409; del buf409  # reuse
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf428, buf429, buf430, buf431, 2432, 122, grid=grid(2432), stream=stream0)
        buf432 = buf413; del buf413  # reuse
        buf433 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf435 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf429, buf430, buf431, primals_388, primals_389, buf432, buf433, buf435, primals_388, primals_389, 128, 19, grid=grid(128), stream=stream0)
        del primals_388
        del primals_389
        buf436 = reinterpret_tensor(buf427, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf427  # reuse
        # Source Nodes: [branch7x7dbl, x_182], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf428, buf432, buf433, primals_69, primals_70, buf436, 295936, grid=grid(295936), stream=stream0)
        del primals_70
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, buf18, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf438 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf437, buf438, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf439 = buf431; del buf431  # reuse
        buf440 = buf430; del buf430  # reuse
        buf441 = buf429; del buf429  # reuse
        # Source Nodes: [x_187], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf438, buf439, buf440, buf441, 2432, 122, grid=grid(2432), stream=stream0)
        buf442 = buf433; del buf433  # reuse
        buf443 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf445 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf439, buf440, buf441, primals_391, primals_392, buf442, buf443, buf445, primals_391, primals_392, 128, 19, grid=grid(128), stream=stream0)
        del primals_391
        del primals_392
        buf446 = reinterpret_tensor(buf437, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf437  # reuse
        # Source Nodes: [branch7x7dbl_1, x_187], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf438, buf442, buf443, primals_71, primals_72, buf446, 295936, grid=grid(295936), stream=stream0)
        del primals_72
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, buf19, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf448 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf447, buf448, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf449 = buf441; del buf441  # reuse
        buf450 = buf440; del buf440  # reuse
        buf451 = buf439; del buf439  # reuse
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf448, buf449, buf450, buf451, 2432, 122, grid=grid(2432), stream=stream0)
        buf452 = buf443; del buf443  # reuse
        buf453 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf455 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf449, buf450, buf451, primals_394, primals_395, buf452, buf453, buf455, primals_394, primals_395, 128, 19, grid=grid(128), stream=stream0)
        del primals_394
        del primals_395
        buf456 = reinterpret_tensor(buf447, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf447  # reuse
        # Source Nodes: [branch7x7dbl_2, x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf448, buf452, buf453, primals_73, primals_74, buf456, 295936, grid=grid(295936), stream=stream0)
        del primals_74
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf456, buf20, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf458 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf457, buf458, 1024, 289, grid=grid(1024, 289), stream=stream0)
        buf459 = buf451; del buf451  # reuse
        buf460 = buf450; del buf450  # reuse
        buf461 = buf449; del buf449  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf458, buf459, buf460, buf461, 2432, 122, grid=grid(2432), stream=stream0)
        buf462 = buf453; del buf453  # reuse
        buf463 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf465 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf459, buf460, buf461, primals_397, primals_398, buf462, buf463, buf465, primals_397, primals_398, 128, 19, grid=grid(128), stream=stream0)
        del buf459
        del buf460
        del buf461
        del primals_397
        del primals_398
        buf466 = reinterpret_tensor(buf457, (8, 128, 17, 17), (36992, 1, 2176, 128), 0); del buf457  # reuse
        # Source Nodes: [branch7x7dbl_3, x_197], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf458, buf462, buf463, primals_75, primals_76, buf466, 295936, grid=grid(295936), stream=stream0)
        del buf463
        del primals_76
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, buf21, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf468 = reinterpret_tensor(buf417, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf417  # reuse
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf467, buf468, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf469 = buf421; del buf421  # reuse
        buf470 = buf420; del buf420  # reuse
        buf471 = buf419; del buf419  # reuse
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf468, buf469, buf470, buf471, 3648, 122, grid=grid(3648), stream=stream0)
        buf472 = buf423; del buf423  # reuse
        buf473 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf475 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf469, buf470, buf471, primals_400, primals_401, buf472, buf473, buf475, primals_400, primals_401, 192, 19, grid=grid(192), stream=stream0)
        del primals_400
        del primals_401
        buf476 = reinterpret_tensor(buf488, (8, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1086 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7dbl_4, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf468, buf472, buf473, primals_77, primals_78, buf476, buf1086, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_78
        buf477 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_89.run(buf386, buf477, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf479 = reinterpret_tensor(buf467, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf467  # reuse
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf478, buf479, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf480 = buf471; del buf471  # reuse
        buf481 = buf470; del buf470  # reuse
        buf482 = buf469; del buf469  # reuse
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf479, buf480, buf481, buf482, 3648, 122, grid=grid(3648), stream=stream0)
        buf483 = buf473; del buf473  # reuse
        buf484 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf486 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf480, buf481, buf482, primals_403, primals_404, buf483, buf484, buf486, primals_403, primals_404, 192, 19, grid=grid(192), stream=stream0)
        del primals_403
        del primals_404
        buf487 = reinterpret_tensor(buf488, (8, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1085 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_8, x_207], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf479, buf483, buf484, primals_79, primals_80, buf487, buf1085, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_80
        buf489 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf488, buf489, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf396
        del buf426
        del buf476
        del buf487
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf491 = reinterpret_tensor(buf478, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf478  # reuse
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf490, buf491, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf492 = buf482; del buf482  # reuse
        buf493 = buf481; del buf481  # reuse
        buf494 = buf480; del buf480  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf491, buf492, buf493, buf494, 3648, 122, grid=grid(3648), stream=stream0)
        buf495 = buf484; del buf484  # reuse
        buf496 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf498 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf492, buf493, buf494, primals_406, primals_407, buf495, buf496, buf498, primals_406, primals_407, 192, 19, grid=grid(192), stream=stream0)
        del primals_406
        del primals_407
        buf591 = buf488; del buf488  # reuse
        buf499 = reinterpret_tensor(buf591, (8, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1084 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_4, x_213], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf491, buf495, buf496, primals_81, primals_82, buf499, buf1084, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_82
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf489, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf501 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf500, buf501, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf502 = empty_strided((1, 160, 1, 1, 19), (3040, 1, 3040, 3040, 160), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((1, 160, 1, 1, 19), (3040, 1, 3040, 3040, 160), device='cuda', dtype=torch.float32)
        buf504 = empty_strided((1, 160, 1, 1, 19), (3040, 1, 3040, 3040, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf501, buf502, buf503, buf504, 3040, 122, grid=grid(3040), stream=stream0)
        buf505 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf506 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf508 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf502, buf503, buf504, primals_409, primals_410, buf505, buf506, buf508, primals_409, primals_410, 160, 19, grid=grid(160), stream=stream0)
        del primals_409
        del primals_410
        buf509 = reinterpret_tensor(buf500, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf500  # reuse
        # Source Nodes: [branch7x7_3, x_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf501, buf505, buf506, primals_83, primals_84, buf509, 369920, grid=grid(369920), stream=stream0)
        del primals_84
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, buf22, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf511 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf510, buf511, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf512 = buf504; del buf504  # reuse
        buf513 = buf503; del buf503  # reuse
        buf514 = buf502; del buf502  # reuse
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf511, buf512, buf513, buf514, 3040, 122, grid=grid(3040), stream=stream0)
        buf515 = buf506; del buf506  # reuse
        buf516 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf518 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf512, buf513, buf514, primals_412, primals_413, buf515, buf516, buf518, primals_412, primals_413, 160, 19, grid=grid(160), stream=stream0)
        del primals_412
        del primals_413
        buf519 = reinterpret_tensor(buf510, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf510  # reuse
        # Source Nodes: [branch7x7_4, x_223], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf511, buf515, buf516, primals_85, primals_86, buf519, 369920, grid=grid(369920), stream=stream0)
        del primals_86
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, buf23, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf521 = reinterpret_tensor(buf490, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf490  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf520, buf521, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf522 = buf494; del buf494  # reuse
        buf523 = buf493; del buf493  # reuse
        buf524 = buf492; del buf492  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf521, buf522, buf523, buf524, 3648, 122, grid=grid(3648), stream=stream0)
        buf525 = buf496; del buf496  # reuse
        buf526 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf528 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf522, buf523, buf524, primals_415, primals_416, buf525, buf526, buf528, primals_415, primals_416, 192, 19, grid=grid(192), stream=stream0)
        del primals_415
        del primals_416
        buf529 = reinterpret_tensor(buf591, (8, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1083 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7_5, x_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf521, buf525, buf526, primals_87, primals_88, buf529, buf1083, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_88
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf489, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf531 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf530, buf531, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf532 = buf514; del buf514  # reuse
        buf533 = buf513; del buf513  # reuse
        buf534 = buf512; del buf512  # reuse
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf531, buf532, buf533, buf534, 3040, 122, grid=grid(3040), stream=stream0)
        buf535 = buf516; del buf516  # reuse
        buf536 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf538 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf532, buf533, buf534, primals_418, primals_419, buf535, buf536, buf538, primals_418, primals_419, 160, 19, grid=grid(160), stream=stream0)
        del primals_418
        del primals_419
        buf539 = reinterpret_tensor(buf530, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf530  # reuse
        # Source Nodes: [branch7x7dbl_5, x_233], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf531, buf535, buf536, primals_89, primals_90, buf539, 369920, grid=grid(369920), stream=stream0)
        del primals_90
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, buf24, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf541 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf540, buf541, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf542 = buf534; del buf534  # reuse
        buf543 = buf533; del buf533  # reuse
        buf544 = buf532; del buf532  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf541, buf542, buf543, buf544, 3040, 122, grid=grid(3040), stream=stream0)
        buf545 = buf536; del buf536  # reuse
        buf546 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf548 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf542, buf543, buf544, primals_421, primals_422, buf545, buf546, buf548, primals_421, primals_422, 160, 19, grid=grid(160), stream=stream0)
        del primals_421
        del primals_422
        buf549 = reinterpret_tensor(buf540, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf540  # reuse
        # Source Nodes: [branch7x7dbl_6, x_238], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf541, buf545, buf546, primals_91, primals_92, buf549, 369920, grid=grid(369920), stream=stream0)
        del primals_92
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, buf25, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf551 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf550, buf551, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf552 = buf544; del buf544  # reuse
        buf553 = buf543; del buf543  # reuse
        buf554 = buf542; del buf542  # reuse
        # Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf551, buf552, buf553, buf554, 3040, 122, grid=grid(3040), stream=stream0)
        buf555 = buf546; del buf546  # reuse
        buf556 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf558 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf552, buf553, buf554, primals_424, primals_425, buf555, buf556, buf558, primals_424, primals_425, 160, 19, grid=grid(160), stream=stream0)
        del primals_424
        del primals_425
        buf559 = reinterpret_tensor(buf550, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf550  # reuse
        # Source Nodes: [branch7x7dbl_7, x_243], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf551, buf555, buf556, primals_93, primals_94, buf559, 369920, grid=grid(369920), stream=stream0)
        del primals_94
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, buf26, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf561 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf560, buf561, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf562 = buf554; del buf554  # reuse
        buf563 = buf553; del buf553  # reuse
        buf564 = buf552; del buf552  # reuse
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf561, buf562, buf563, buf564, 3040, 122, grid=grid(3040), stream=stream0)
        buf565 = buf556; del buf556  # reuse
        buf566 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf568 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf562, buf563, buf564, primals_427, primals_428, buf565, buf566, buf568, primals_427, primals_428, 160, 19, grid=grid(160), stream=stream0)
        del primals_427
        del primals_428
        buf569 = reinterpret_tensor(buf560, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf560  # reuse
        # Source Nodes: [branch7x7dbl_8, x_248], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf561, buf565, buf566, primals_95, primals_96, buf569, 369920, grid=grid(369920), stream=stream0)
        del primals_96
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, buf27, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf571 = reinterpret_tensor(buf520, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf520  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf570, buf571, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf572 = buf524; del buf524  # reuse
        buf573 = buf523; del buf523  # reuse
        buf574 = buf522; del buf522  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf571, buf572, buf573, buf574, 3648, 122, grid=grid(3648), stream=stream0)
        buf575 = buf526; del buf526  # reuse
        buf576 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf578 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf572, buf573, buf574, primals_430, primals_431, buf575, buf576, buf578, primals_430, primals_431, 192, 19, grid=grid(192), stream=stream0)
        del primals_430
        del primals_431
        buf579 = reinterpret_tensor(buf591, (8, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1082 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7dbl_9, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf571, buf575, buf576, primals_97, primals_98, buf579, buf1082, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_98
        buf580 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_89.run(buf489, buf580, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf582 = reinterpret_tensor(buf570, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf570  # reuse
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf581, buf582, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf583 = buf574; del buf574  # reuse
        buf584 = buf573; del buf573  # reuse
        buf585 = buf572; del buf572  # reuse
        # Source Nodes: [x_258], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf582, buf583, buf584, buf585, 3648, 122, grid=grid(3648), stream=stream0)
        buf586 = buf576; del buf576  # reuse
        buf587 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf589 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf583, buf584, buf585, primals_433, primals_434, buf586, buf587, buf589, primals_433, primals_434, 192, 19, grid=grid(192), stream=stream0)
        del primals_433
        del primals_434
        buf590 = reinterpret_tensor(buf591, (8, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1081 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_10, x_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf582, buf586, buf587, primals_99, primals_100, buf590, buf1081, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_100
        buf592 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf591, buf592, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf499
        del buf529
        del buf579
        del buf590
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_239, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf594 = reinterpret_tensor(buf581, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf581  # reuse
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf593, buf594, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf595 = buf585; del buf585  # reuse
        buf596 = buf584; del buf584  # reuse
        buf597 = buf583; del buf583  # reuse
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf594, buf595, buf596, buf597, 3648, 122, grid=grid(3648), stream=stream0)
        buf598 = buf587; del buf587  # reuse
        buf599 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf601 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf595, buf596, buf597, primals_436, primals_437, buf598, buf599, buf601, primals_436, primals_437, 192, 19, grid=grid(192), stream=stream0)
        del primals_436
        del primals_437
        buf694 = buf591; del buf591  # reuse
        buf602 = reinterpret_tensor(buf694, (8, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1080 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_5, x_264], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf594, buf598, buf599, primals_101, primals_102, buf602, buf1080, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_102
        # Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf592, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf604 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf603, buf604, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf605 = buf564; del buf564  # reuse
        buf606 = buf563; del buf563  # reuse
        buf607 = buf562; del buf562  # reuse
        # Source Nodes: [x_269], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf604, buf605, buf606, buf607, 3040, 122, grid=grid(3040), stream=stream0)
        buf608 = buf566; del buf566  # reuse
        buf609 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf611 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf605, buf606, buf607, primals_439, primals_440, buf608, buf609, buf611, primals_439, primals_440, 160, 19, grid=grid(160), stream=stream0)
        del primals_439
        del primals_440
        buf612 = reinterpret_tensor(buf603, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf603  # reuse
        # Source Nodes: [branch7x7_6, x_269], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf604, buf608, buf609, primals_103, primals_104, buf612, 369920, grid=grid(369920), stream=stream0)
        del primals_104
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, buf28, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf614 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf613, buf614, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf615 = buf607; del buf607  # reuse
        buf616 = buf606; del buf606  # reuse
        buf617 = buf605; del buf605  # reuse
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf614, buf615, buf616, buf617, 3040, 122, grid=grid(3040), stream=stream0)
        buf618 = buf609; del buf609  # reuse
        buf619 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf621 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf615, buf616, buf617, primals_442, primals_443, buf618, buf619, buf621, primals_442, primals_443, 160, 19, grid=grid(160), stream=stream0)
        del primals_442
        del primals_443
        buf622 = reinterpret_tensor(buf613, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf613  # reuse
        # Source Nodes: [branch7x7_7, x_274], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf614, buf618, buf619, primals_105, primals_106, buf622, 369920, grid=grid(369920), stream=stream0)
        del primals_106
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, buf29, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf624 = reinterpret_tensor(buf593, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf593  # reuse
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf623, buf624, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf625 = buf597; del buf597  # reuse
        buf626 = buf596; del buf596  # reuse
        buf627 = buf595; del buf595  # reuse
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf624, buf625, buf626, buf627, 3648, 122, grid=grid(3648), stream=stream0)
        buf628 = buf599; del buf599  # reuse
        buf629 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf631 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf625, buf626, buf627, primals_445, primals_446, buf628, buf629, buf631, primals_445, primals_446, 192, 19, grid=grid(192), stream=stream0)
        del primals_445
        del primals_446
        buf632 = reinterpret_tensor(buf694, (8, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1079 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7_8, x_279], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf624, buf628, buf629, primals_107, primals_108, buf632, buf1079, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_108
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf592, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf634 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf633, buf634, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf635 = buf617; del buf617  # reuse
        buf636 = buf616; del buf616  # reuse
        buf637 = buf615; del buf615  # reuse
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf634, buf635, buf636, buf637, 3040, 122, grid=grid(3040), stream=stream0)
        buf638 = buf619; del buf619  # reuse
        buf639 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf641 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf635, buf636, buf637, primals_448, primals_449, buf638, buf639, buf641, primals_448, primals_449, 160, 19, grid=grid(160), stream=stream0)
        del primals_448
        del primals_449
        buf642 = reinterpret_tensor(buf633, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf633  # reuse
        # Source Nodes: [branch7x7dbl_10, x_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf634, buf638, buf639, primals_109, primals_110, buf642, 369920, grid=grid(369920), stream=stream0)
        del primals_110
        # Source Nodes: [x_288], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, buf30, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf644 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_288], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf643, buf644, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf645 = buf637; del buf637  # reuse
        buf646 = buf636; del buf636  # reuse
        buf647 = buf635; del buf635  # reuse
        # Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf644, buf645, buf646, buf647, 3040, 122, grid=grid(3040), stream=stream0)
        buf648 = buf639; del buf639  # reuse
        buf649 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf651 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf645, buf646, buf647, primals_451, primals_452, buf648, buf649, buf651, primals_451, primals_452, 160, 19, grid=grid(160), stream=stream0)
        del primals_451
        del primals_452
        buf652 = reinterpret_tensor(buf643, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf643  # reuse
        # Source Nodes: [branch7x7dbl_11, x_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf644, buf648, buf649, primals_111, primals_112, buf652, 369920, grid=grid(369920), stream=stream0)
        del primals_112
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf653 = extern_kernels.convolution(buf652, buf31, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf653, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf654 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf653, buf654, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf655 = buf647; del buf647  # reuse
        buf656 = buf646; del buf646  # reuse
        buf657 = buf645; del buf645  # reuse
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf654, buf655, buf656, buf657, 3040, 122, grid=grid(3040), stream=stream0)
        buf658 = buf649; del buf649  # reuse
        buf659 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf661 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf655, buf656, buf657, primals_454, primals_455, buf658, buf659, buf661, primals_454, primals_455, 160, 19, grid=grid(160), stream=stream0)
        del primals_454
        del primals_455
        buf662 = reinterpret_tensor(buf653, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf653  # reuse
        # Source Nodes: [branch7x7dbl_12, x_294], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf654, buf658, buf659, primals_113, primals_114, buf662, 369920, grid=grid(369920), stream=stream0)
        del primals_114
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, buf32, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf664 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf663, buf664, 1280, 289, grid=grid(1280, 289), stream=stream0)
        buf665 = buf657; del buf657  # reuse
        buf666 = buf656; del buf656  # reuse
        buf667 = buf655; del buf655  # reuse
        # Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf664, buf665, buf666, buf667, 3040, 122, grid=grid(3040), stream=stream0)
        buf668 = buf659; del buf659  # reuse
        buf669 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf671 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf665, buf666, buf667, primals_457, primals_458, buf668, buf669, buf671, primals_457, primals_458, 160, 19, grid=grid(160), stream=stream0)
        del buf665
        del buf666
        del buf667
        del primals_457
        del primals_458
        buf672 = reinterpret_tensor(buf663, (8, 160, 17, 17), (46240, 1, 2720, 160), 0); del buf663  # reuse
        # Source Nodes: [branch7x7dbl_13, x_299], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_93.run(buf664, buf668, buf669, primals_115, primals_116, buf672, 369920, grid=grid(369920), stream=stream0)
        del buf669
        del primals_116
        # Source Nodes: [x_303], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, buf33, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf674 = reinterpret_tensor(buf623, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf623  # reuse
        # Source Nodes: [x_303], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf673, buf674, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf675 = buf627; del buf627  # reuse
        buf676 = buf626; del buf626  # reuse
        buf677 = buf625; del buf625  # reuse
        # Source Nodes: [x_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf674, buf675, buf676, buf677, 3648, 122, grid=grid(3648), stream=stream0)
        buf678 = buf629; del buf629  # reuse
        buf679 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf681 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf675, buf676, buf677, primals_460, primals_461, buf678, buf679, buf681, primals_460, primals_461, 192, 19, grid=grid(192), stream=stream0)
        del primals_460
        del primals_461
        buf682 = reinterpret_tensor(buf694, (8, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1078 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7dbl_14, x_304], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf674, buf678, buf679, primals_117, primals_118, buf682, buf1078, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_118
        buf683 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_89.run(buf592, buf683, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf685 = reinterpret_tensor(buf673, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf673  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf684, buf685, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf686 = buf677; del buf677  # reuse
        buf687 = buf676; del buf676  # reuse
        buf688 = buf675; del buf675  # reuse
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf685, buf686, buf687, buf688, 3648, 122, grid=grid(3648), stream=stream0)
        buf689 = buf679; del buf679  # reuse
        buf690 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf692 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf686, buf687, buf688, primals_463, primals_464, buf689, buf690, buf692, primals_463, primals_464, 192, 19, grid=grid(192), stream=stream0)
        del primals_463
        del primals_464
        buf693 = reinterpret_tensor(buf694, (8, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1077 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_12, x_309], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf685, buf689, buf690, primals_119, primals_120, buf693, buf1077, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_120
        buf695 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf694, buf695, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf602
        del buf632
        del buf682
        del buf693
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf696 = extern_kernels.convolution(buf695, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf696, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf697 = reinterpret_tensor(buf684, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf684  # reuse
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf696, buf697, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf698 = buf688; del buf688  # reuse
        buf699 = buf687; del buf687  # reuse
        buf700 = buf686; del buf686  # reuse
        # Source Nodes: [x_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf697, buf698, buf699, buf700, 3648, 122, grid=grid(3648), stream=stream0)
        buf701 = buf690; del buf690  # reuse
        buf702 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf704 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf698, buf699, buf700, primals_466, primals_467, buf701, buf702, buf704, primals_466, primals_467, 192, 19, grid=grid(192), stream=stream0)
        del primals_466
        del primals_467
        buf797 = buf694; del buf694  # reuse
        buf705 = reinterpret_tensor(buf797, (8, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1076 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_6, x_315], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf697, buf701, buf702, primals_121, primals_122, buf705, buf1076, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_122
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf695, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf707 = reinterpret_tensor(buf696, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf696  # reuse
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf706, buf707, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf708 = buf700; del buf700  # reuse
        buf709 = buf699; del buf699  # reuse
        buf710 = buf698; del buf698  # reuse
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf707, buf708, buf709, buf710, 3648, 122, grid=grid(3648), stream=stream0)
        buf711 = buf702; del buf702  # reuse
        buf712 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf714 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf708, buf709, buf710, primals_469, primals_470, buf711, buf712, buf714, primals_469, primals_470, 192, 19, grid=grid(192), stream=stream0)
        del primals_469
        del primals_470
        buf715 = reinterpret_tensor(buf706, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf706  # reuse
        # Source Nodes: [branch7x7_9, x_320], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf707, buf711, buf712, primals_123, primals_124, buf715, 443904, grid=grid(443904), stream=stream0)
        del primals_124
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf716 = extern_kernels.convolution(buf715, buf34, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf716, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf717 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf716, buf717, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf718 = buf710; del buf710  # reuse
        buf719 = buf709; del buf709  # reuse
        buf720 = buf708; del buf708  # reuse
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf717, buf718, buf719, buf720, 3648, 122, grid=grid(3648), stream=stream0)
        buf721 = buf712; del buf712  # reuse
        buf722 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf724 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf718, buf719, buf720, primals_472, primals_473, buf721, buf722, buf724, primals_472, primals_473, 192, 19, grid=grid(192), stream=stream0)
        del primals_472
        del primals_473
        buf725 = reinterpret_tensor(buf716, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf716  # reuse
        # Source Nodes: [branch7x7_10, x_325], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf717, buf721, buf722, primals_125, primals_126, buf725, 443904, grid=grid(443904), stream=stream0)
        del primals_126
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        buf726 = extern_kernels.convolution(buf725, buf35, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf726, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf727 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf726, buf727, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf728 = buf720; del buf720  # reuse
        buf729 = buf719; del buf719  # reuse
        buf730 = buf718; del buf718  # reuse
        # Source Nodes: [x_330], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf727, buf728, buf729, buf730, 3648, 122, grid=grid(3648), stream=stream0)
        buf731 = buf722; del buf722  # reuse
        buf732 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf734 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_330], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf728, buf729, buf730, primals_475, primals_476, buf731, buf732, buf734, primals_475, primals_476, 192, 19, grid=grid(192), stream=stream0)
        del primals_475
        del primals_476
        buf735 = reinterpret_tensor(buf797, (8, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1075 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7_11, x_330], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf727, buf731, buf732, primals_127, primals_128, buf735, buf1075, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_128
        # Source Nodes: [x_334], Original ATen: [aten.convolution]
        buf736 = extern_kernels.convolution(buf695, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf736, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf737 = reinterpret_tensor(buf726, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf726  # reuse
        # Source Nodes: [x_334], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf736, buf737, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf738 = buf730; del buf730  # reuse
        buf739 = buf729; del buf729  # reuse
        buf740 = buf728; del buf728  # reuse
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf737, buf738, buf739, buf740, 3648, 122, grid=grid(3648), stream=stream0)
        buf741 = buf732; del buf732  # reuse
        buf742 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf744 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf738, buf739, buf740, primals_478, primals_479, buf741, buf742, buf744, primals_478, primals_479, 192, 19, grid=grid(192), stream=stream0)
        del primals_478
        del primals_479
        buf745 = reinterpret_tensor(buf736, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf736  # reuse
        # Source Nodes: [branch7x7dbl_15, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf737, buf741, buf742, primals_129, primals_130, buf745, 443904, grid=grid(443904), stream=stream0)
        del primals_130
        # Source Nodes: [x_339], Original ATen: [aten.convolution]
        buf746 = extern_kernels.convolution(buf745, buf36, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf746, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf747 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf746, buf747, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf748 = buf740; del buf740  # reuse
        buf749 = buf739; del buf739  # reuse
        buf750 = buf738; del buf738  # reuse
        # Source Nodes: [x_340], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf747, buf748, buf749, buf750, 3648, 122, grid=grid(3648), stream=stream0)
        buf751 = buf742; del buf742  # reuse
        buf752 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf754 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf748, buf749, buf750, primals_481, primals_482, buf751, buf752, buf754, primals_481, primals_482, 192, 19, grid=grid(192), stream=stream0)
        del primals_481
        del primals_482
        buf755 = reinterpret_tensor(buf746, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf746  # reuse
        # Source Nodes: [branch7x7dbl_16, x_340], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf747, buf751, buf752, primals_131, primals_132, buf755, 443904, grid=grid(443904), stream=stream0)
        del primals_132
        # Source Nodes: [x_344], Original ATen: [aten.convolution]
        buf756 = extern_kernels.convolution(buf755, buf37, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf756, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf757 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_344], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf756, buf757, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf758 = buf750; del buf750  # reuse
        buf759 = buf749; del buf749  # reuse
        buf760 = buf748; del buf748  # reuse
        # Source Nodes: [x_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf757, buf758, buf759, buf760, 3648, 122, grid=grid(3648), stream=stream0)
        buf761 = buf752; del buf752  # reuse
        buf762 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf764 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf758, buf759, buf760, primals_484, primals_485, buf761, buf762, buf764, primals_484, primals_485, 192, 19, grid=grid(192), stream=stream0)
        del primals_484
        del primals_485
        buf765 = reinterpret_tensor(buf756, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf756  # reuse
        # Source Nodes: [branch7x7dbl_17, x_345], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf757, buf761, buf762, primals_133, primals_134, buf765, 443904, grid=grid(443904), stream=stream0)
        del primals_134
        # Source Nodes: [x_349], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf765, buf38, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf767 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_349], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf766, buf767, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf768 = buf760; del buf760  # reuse
        buf769 = buf759; del buf759  # reuse
        buf770 = buf758; del buf758  # reuse
        # Source Nodes: [x_350], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf767, buf768, buf769, buf770, 3648, 122, grid=grid(3648), stream=stream0)
        buf771 = buf762; del buf762  # reuse
        buf772 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf774 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf768, buf769, buf770, primals_487, primals_488, buf771, buf772, buf774, primals_487, primals_488, 192, 19, grid=grid(192), stream=stream0)
        del primals_487
        del primals_488
        buf775 = reinterpret_tensor(buf766, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf766  # reuse
        # Source Nodes: [branch7x7dbl_18, x_350], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf767, buf771, buf772, primals_135, primals_136, buf775, 443904, grid=grid(443904), stream=stream0)
        del primals_136
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, buf39, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf776, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf777 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf776, buf777, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf778 = buf770; del buf770  # reuse
        buf779 = buf769; del buf769  # reuse
        buf780 = buf768; del buf768  # reuse
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf777, buf778, buf779, buf780, 3648, 122, grid=grid(3648), stream=stream0)
        buf781 = buf772; del buf772  # reuse
        buf782 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf784 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf778, buf779, buf780, primals_490, primals_491, buf781, buf782, buf784, primals_490, primals_491, 192, 19, grid=grid(192), stream=stream0)
        del primals_490
        del primals_491
        buf785 = reinterpret_tensor(buf797, (8, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1074 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7dbl_19, x_355], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf777, buf781, buf782, primals_137, primals_138, buf785, buf1074, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_138
        buf786 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_89.run(buf695, buf786, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf787 = extern_kernels.convolution(buf786, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf787, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf788 = reinterpret_tensor(buf776, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf776  # reuse
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf787, buf788, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf789 = buf780; del buf780  # reuse
        buf790 = buf779; del buf779  # reuse
        buf791 = buf778; del buf778  # reuse
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf788, buf789, buf790, buf791, 3648, 122, grid=grid(3648), stream=stream0)
        buf792 = buf782; del buf782  # reuse
        buf793 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf795 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf789, buf790, buf791, primals_493, primals_494, buf792, buf793, buf795, primals_493, primals_494, 192, 19, grid=grid(192), stream=stream0)
        del primals_493
        del primals_494
        buf796 = reinterpret_tensor(buf797, (8, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1073 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_14, x_360], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf788, buf792, buf793, primals_139, primals_140, buf796, buf1073, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del primals_140
        buf798 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf797, buf798, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf705
        del buf735
        del buf785
        del buf796
        del buf797
        # Source Nodes: [x_366], Original ATen: [aten.convolution]
        buf799 = extern_kernels.convolution(buf798, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf799, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf800 = reinterpret_tensor(buf787, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf787  # reuse
        # Source Nodes: [x_366], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf799, buf800, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf801 = buf791; del buf791  # reuse
        buf802 = buf790; del buf790  # reuse
        buf803 = buf789; del buf789  # reuse
        # Source Nodes: [x_367], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf800, buf801, buf802, buf803, 3648, 122, grid=grid(3648), stream=stream0)
        buf804 = buf793; del buf793  # reuse
        buf805 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf807 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_367], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf801, buf802, buf803, primals_496, primals_497, buf804, buf805, buf807, primals_496, primals_497, 192, 19, grid=grid(192), stream=stream0)
        del primals_496
        del primals_497
        buf808 = reinterpret_tensor(buf799, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf799  # reuse
        # Source Nodes: [branch3x3_1, x_367], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf800, buf804, buf805, primals_141, primals_142, buf808, 443904, grid=grid(443904), stream=stream0)
        del primals_142
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf809 = extern_kernels.convolution(buf808, buf40, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf809, (8, 320, 8, 8), (20480, 64, 8, 1))
        buf810 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf809, buf810, 2560, 64, grid=grid(2560, 64), stream=stream0)
        buf811 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf812 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf813 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_96.run(buf810, buf811, buf812, buf813, 1280, 128, grid=grid(1280), stream=stream0)
        buf814 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf815 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf817 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_97.run(buf811, buf812, buf813, primals_499, primals_500, buf814, buf815, buf817, primals_499, primals_500, 320, 4, grid=grid(320), stream=stream0)
        del primals_499
        del primals_500
        buf861 = empty((8, 1280, 8, 8), device='cuda', dtype=torch.float32)
        buf818 = reinterpret_tensor(buf861, (8, 320, 8, 8), (81920, 64, 8, 1), 0)  # alias
        buf1072 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch3x3_2, x_372], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_98.run(buf810, buf814, buf815, primals_143, primals_144, buf818, buf1072, 2560, 64, grid=grid(2560, 64), stream=stream0)
        del primals_144
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        buf819 = extern_kernels.convolution(buf798, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf819, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf820 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf819, buf820, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf821 = buf803; del buf803  # reuse
        buf822 = buf802; del buf802  # reuse
        buf823 = buf801; del buf801  # reuse
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf820, buf821, buf822, buf823, 3648, 122, grid=grid(3648), stream=stream0)
        buf824 = buf805; del buf805  # reuse
        buf825 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf827 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf821, buf822, buf823, primals_502, primals_503, buf824, buf825, buf827, primals_502, primals_503, 192, 19, grid=grid(192), stream=stream0)
        del primals_502
        del primals_503
        buf828 = reinterpret_tensor(buf819, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf819  # reuse
        # Source Nodes: [branch7x7x3, x_377], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf820, buf824, buf825, primals_145, primals_146, buf828, 443904, grid=grid(443904), stream=stream0)
        del primals_146
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        buf829 = extern_kernels.convolution(buf828, buf41, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf829, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf830 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf829, buf830, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf831 = buf823; del buf823  # reuse
        buf832 = buf822; del buf822  # reuse
        buf833 = buf821; del buf821  # reuse
        # Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf830, buf831, buf832, buf833, 3648, 122, grid=grid(3648), stream=stream0)
        buf834 = buf825; del buf825  # reuse
        buf835 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf837 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf831, buf832, buf833, primals_505, primals_506, buf834, buf835, buf837, primals_505, primals_506, 192, 19, grid=grid(192), stream=stream0)
        del primals_505
        del primals_506
        buf838 = reinterpret_tensor(buf829, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf829  # reuse
        # Source Nodes: [branch7x7x3_1, x_382], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf830, buf834, buf835, primals_147, primals_148, buf838, 443904, grid=grid(443904), stream=stream0)
        del primals_148
        # Source Nodes: [x_386], Original ATen: [aten.convolution]
        buf839 = extern_kernels.convolution(buf838, buf42, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf839, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf840 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf839, buf840, 1536, 289, grid=grid(1536, 289), stream=stream0)
        buf841 = buf833; del buf833  # reuse
        buf842 = buf832; del buf832  # reuse
        buf843 = buf831; del buf831  # reuse
        # Source Nodes: [x_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf840, buf841, buf842, buf843, 3648, 122, grid=grid(3648), stream=stream0)
        buf844 = buf835; del buf835  # reuse
        buf845 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf847 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf841, buf842, buf843, primals_508, primals_509, buf844, buf845, buf847, primals_508, primals_509, 192, 19, grid=grid(192), stream=stream0)
        del buf841
        del buf842
        del buf843
        del primals_508
        del primals_509
        buf848 = reinterpret_tensor(buf839, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf839  # reuse
        # Source Nodes: [branch7x7x3_2, x_387], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_94.run(buf840, buf844, buf845, primals_149, primals_150, buf848, 443904, grid=grid(443904), stream=stream0)
        del primals_150
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf849 = extern_kernels.convolution(buf848, buf43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf849, (8, 192, 8, 8), (12288, 64, 8, 1))
        buf850 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf849, buf850, 1536, 64, grid=grid(1536, 64), stream=stream0)
        buf851 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf852 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf853 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf850, buf851, buf852, buf853, 768, 128, grid=grid(768), stream=stream0)
        buf854 = buf845; del buf845  # reuse
        buf855 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf857 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf851, buf852, buf853, primals_511, primals_512, buf854, buf855, buf857, primals_511, primals_512, 192, 4, grid=grid(192), stream=stream0)
        del primals_511
        del primals_512
        buf858 = reinterpret_tensor(buf861, (8, 192, 8, 8), (81920, 64, 8, 1), 20480)  # alias
        buf1071 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch7x7x3_3, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_102.run(buf850, buf854, buf855, primals_151, primals_152, buf858, buf1071, 1536, 64, grid=grid(1536, 64), stream=stream0)
        del primals_152
        buf859 = reinterpret_tensor(buf861, (8, 768, 8, 8), (81920, 64, 8, 1), 32768)  # alias
        # Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_103.run(buf798, buf859, 6144, 64, grid=grid(6144, 64), stream=stream0)
        buf860 = empty_strided((8, 768, 8, 8), (49152, 1, 6144, 768), device='cuda', dtype=torch.int64)
        # Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_104.run(buf798, buf860, 393216, grid=grid(393216), stream=stream0)
        buf862 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf861, buf862, 10240, 64, grid=grid(10240, 64), stream=stream0)
        del buf818
        del buf858
        del buf859
        # Source Nodes: [x_397], Original ATen: [aten.convolution]
        buf863 = extern_kernels.convolution(buf862, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf863, (8, 320, 8, 8), (20480, 64, 8, 1))
        buf864 = reinterpret_tensor(buf809, (8, 320, 8, 8), (20480, 1, 2560, 320), 0); del buf809  # reuse
        # Source Nodes: [x_397], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf863, buf864, 2560, 64, grid=grid(2560, 64), stream=stream0)
        buf865 = buf813; del buf813  # reuse
        buf866 = buf812; del buf812  # reuse
        buf867 = buf811; del buf811  # reuse
        # Source Nodes: [x_398], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_96.run(buf864, buf865, buf866, buf867, 1280, 128, grid=grid(1280), stream=stream0)
        buf868 = buf815; del buf815  # reuse
        buf869 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf871 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_97.run(buf865, buf866, buf867, primals_514, primals_515, buf868, buf869, buf871, primals_514, primals_515, 320, 4, grid=grid(320), stream=stream0)
        del primals_514
        del primals_515
        buf958 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        buf872 = reinterpret_tensor(buf958, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        buf1070 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_7, x_398], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_106.run(buf864, buf868, buf869, primals_153, primals_154, buf872, buf1070, 2560, 64, grid=grid(2560, 64), stream=stream0)
        del primals_154
        # Source Nodes: [x_402], Original ATen: [aten.convolution]
        buf873 = extern_kernels.convolution(buf862, primals_266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf873, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf874 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_402], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf873, buf874, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf875 = empty_strided((1, 384, 1, 1, 4), (1536, 1, 1536, 1536, 384), device='cuda', dtype=torch.float32)
        buf876 = empty_strided((1, 384, 1, 1, 4), (1536, 1, 1536, 1536, 384), device='cuda', dtype=torch.float32)
        buf877 = empty_strided((1, 384, 1, 1, 4), (1536, 1, 1536, 1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_403], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf874, buf875, buf876, buf877, 1536, 128, grid=grid(1536), stream=stream0)
        buf878 = buf349; del buf349  # reuse
        buf879 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf881 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_403], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf875, buf876, buf877, primals_517, primals_518, buf878, buf879, buf881, primals_517, primals_518, 384, 4, grid=grid(384), stream=stream0)
        del primals_517
        del primals_518
        buf882 = reinterpret_tensor(buf873, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf873  # reuse
        # Source Nodes: [branch3x3_3, x_403], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_110.run(buf874, buf878, buf879, primals_155, primals_156, buf882, 196608, grid=grid(196608), stream=stream0)
        del primals_156
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, buf44, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf884 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf883, buf884, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf885 = buf877; del buf877  # reuse
        buf886 = buf876; del buf876  # reuse
        buf887 = buf875; del buf875  # reuse
        # Source Nodes: [x_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf884, buf885, buf886, buf887, 1536, 128, grid=grid(1536), stream=stream0)
        buf888 = buf879; del buf879  # reuse
        buf889 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf891 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf885, buf886, buf887, primals_520, primals_521, buf888, buf889, buf891, primals_520, primals_521, 384, 4, grid=grid(384), stream=stream0)
        del primals_520
        del primals_521
        buf903 = empty((8, 768, 8, 8), device='cuda', dtype=torch.float32)
        buf892 = reinterpret_tensor(buf903, (8, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1069 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_408, x_411], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf884, buf888, buf889, primals_157, primals_158, buf892, buf1069, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_158
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        buf893 = extern_kernels.convolution(buf882, buf45, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf893, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf894 = reinterpret_tensor(buf883, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf883  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf893, buf894, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf895 = buf887; del buf887  # reuse
        buf896 = buf886; del buf886  # reuse
        buf897 = buf885; del buf885  # reuse
        # Source Nodes: [x_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf894, buf895, buf896, buf897, 1536, 128, grid=grid(1536), stream=stream0)
        buf898 = buf889; del buf889  # reuse
        buf899 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf901 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf895, buf896, buf897, primals_523, primals_524, buf898, buf899, buf901, primals_523, primals_524, 384, 4, grid=grid(384), stream=stream0)
        del primals_523
        del primals_524
        buf902 = reinterpret_tensor(buf903, (8, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1068 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_413, x_416], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf894, buf898, buf899, primals_159, primals_160, buf902, buf1068, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_160
        del buf892
        del buf902
        # Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf904 = extern_kernels.convolution(buf862, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf904, (8, 448, 8, 8), (28672, 64, 8, 1))
        buf905 = empty_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_112.run(buf904, buf905, 3584, 64, grid=grid(3584, 64), stream=stream0)
        buf906 = empty_strided((1, 448, 1, 1, 4), (1792, 1, 1792, 1792, 448), device='cuda', dtype=torch.float32)
        buf907 = empty_strided((1, 448, 1, 1, 4), (1792, 1, 1792, 1792, 448), device='cuda', dtype=torch.float32)
        buf908 = empty_strided((1, 448, 1, 1, 4), (1792, 1, 1792, 1792, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_113.run(buf905, buf906, buf907, buf908, 1792, 128, grid=grid(1792), stream=stream0)
        buf909 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf910 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf912 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_114.run(buf906, buf907, buf908, primals_526, primals_527, buf909, buf910, buf912, primals_526, primals_527, 448, 4, grid=grid(448), stream=stream0)
        del primals_526
        del primals_527
        buf913 = reinterpret_tensor(buf904, (8, 448, 8, 8), (28672, 1, 3584, 448), 0); del buf904  # reuse
        # Source Nodes: [branch3x3dbl_12, x_418], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_115.run(buf905, buf909, buf910, primals_161, primals_162, buf913, 229376, grid=grid(229376), stream=stream0)
        del primals_162
        # Source Nodes: [x_422], Original ATen: [aten.convolution]
        buf914 = extern_kernels.convolution(buf913, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf914, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf915 = reinterpret_tensor(buf893, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf893  # reuse
        # Source Nodes: [x_422], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf914, buf915, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf916 = buf897; del buf897  # reuse
        buf917 = buf896; del buf896  # reuse
        buf918 = buf895; del buf895  # reuse
        # Source Nodes: [x_423], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf915, buf916, buf917, buf918, 1536, 128, grid=grid(1536), stream=stream0)
        buf919 = buf899; del buf899  # reuse
        buf920 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf922 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_423], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf916, buf917, buf918, primals_529, primals_530, buf919, buf920, buf922, primals_529, primals_530, 384, 4, grid=grid(384), stream=stream0)
        del primals_529
        del primals_530
        buf923 = reinterpret_tensor(buf914, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf914  # reuse
        # Source Nodes: [branch3x3dbl_13, x_423], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_110.run(buf915, buf919, buf920, primals_163, primals_164, buf923, 196608, grid=grid(196608), stream=stream0)
        del primals_164
        # Source Nodes: [x_427], Original ATen: [aten.convolution]
        buf924 = extern_kernels.convolution(buf923, buf47, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf924, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf925 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_427], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf924, buf925, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf926 = buf918; del buf918  # reuse
        buf927 = buf917; del buf917  # reuse
        buf928 = buf916; del buf916  # reuse
        # Source Nodes: [x_428], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf925, buf926, buf927, buf928, 1536, 128, grid=grid(1536), stream=stream0)
        buf929 = buf920; del buf920  # reuse
        buf930 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf932 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_428], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf926, buf927, buf928, primals_532, primals_533, buf929, buf930, buf932, primals_532, primals_533, 384, 4, grid=grid(384), stream=stream0)
        del primals_532
        del primals_533
        buf944 = empty((8, 768, 8, 8), device='cuda', dtype=torch.float32)
        buf933 = reinterpret_tensor(buf944, (8, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1067 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_428, x_431], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf925, buf929, buf930, primals_165, primals_166, buf933, buf1067, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_166
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf934 = extern_kernels.convolution(buf923, buf48, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf934, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf935 = reinterpret_tensor(buf924, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf924  # reuse
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf934, buf935, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf936 = buf928; del buf928  # reuse
        buf937 = buf927; del buf927  # reuse
        buf938 = buf926; del buf926  # reuse
        # Source Nodes: [x_433], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf935, buf936, buf937, buf938, 1536, 128, grid=grid(1536), stream=stream0)
        buf939 = buf930; del buf930  # reuse
        buf940 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf942 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_433], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf936, buf937, buf938, primals_535, primals_536, buf939, buf940, buf942, primals_535, primals_536, 384, 4, grid=grid(384), stream=stream0)
        del primals_535
        del primals_536
        buf943 = reinterpret_tensor(buf944, (8, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1066 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_433, x_436], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf935, buf939, buf940, primals_167, primals_168, buf943, buf1066, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_168
        buf945 = reinterpret_tensor(buf861, (8, 1280, 8, 8), (81920, 1, 10240, 1280), 0); del buf861  # reuse
        # Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_116.run(buf862, buf945, 655360, grid=grid(655360), stream=stream0)
        del buf933
        del buf943
        # Source Nodes: [x_437], Original ATen: [aten.convolution]
        buf946 = extern_kernels.convolution(buf945, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf946, (8, 192, 8, 8), (12288, 64, 8, 1))
        buf947 = reinterpret_tensor(buf849, (8, 192, 8, 8), (12288, 1, 1536, 192), 0); del buf849  # reuse
        # Source Nodes: [x_437], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf946, buf947, 1536, 64, grid=grid(1536, 64), stream=stream0)
        buf948 = buf853; del buf853  # reuse
        buf949 = buf852; del buf852  # reuse
        buf950 = buf851; del buf851  # reuse
        # Source Nodes: [x_438], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf947, buf948, buf949, buf950, 768, 128, grid=grid(768), stream=stream0)
        buf951 = buf855; del buf855  # reuse
        buf952 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf954 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf948, buf949, buf950, primals_538, primals_539, buf951, buf952, buf954, primals_538, primals_539, 192, 4, grid=grid(192), stream=stream0)
        del primals_538
        del primals_539
        buf955 = reinterpret_tensor(buf958, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        buf1065 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_17, x_438], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_117.run(buf947, buf951, buf952, primals_169, primals_170, buf955, buf1065, 1536, 64, grid=grid(1536, 64), stream=stream0)
        del primals_170
        buf956 = reinterpret_tensor(buf958, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_118.run(buf903, buf956, 393216, grid=grid(393216), stream=stream0)
        buf957 = reinterpret_tensor(buf958, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_118.run(buf944, buf957, 393216, grid=grid(393216), stream=stream0)
        buf959 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_119.run(buf958, buf959, 16384, 64, grid=grid(16384, 64), stream=stream0)
        del buf872
        del buf955
        del buf956
        del buf957
        # Source Nodes: [x_443], Original ATen: [aten.convolution]
        buf960 = extern_kernels.convolution(buf959, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf960, (8, 320, 8, 8), (20480, 64, 8, 1))
        buf961 = reinterpret_tensor(buf863, (8, 320, 8, 8), (20480, 1, 2560, 320), 0); del buf863  # reuse
        # Source Nodes: [x_443], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf960, buf961, 2560, 64, grid=grid(2560, 64), stream=stream0)
        del buf960
        buf962 = buf867; del buf867  # reuse
        buf963 = buf866; del buf866  # reuse
        buf964 = buf865; del buf865  # reuse
        # Source Nodes: [x_444], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_96.run(buf961, buf962, buf963, buf964, 1280, 128, grid=grid(1280), stream=stream0)
        buf965 = buf869; del buf869  # reuse
        buf966 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf968 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_444], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_97.run(buf962, buf963, buf964, primals_541, primals_542, buf965, buf966, buf968, primals_541, primals_542, 320, 4, grid=grid(320), stream=stream0)
        del buf962
        del buf963
        del buf964
        del primals_541
        del primals_542
        buf1055 = buf958; del buf958  # reuse
        buf969 = reinterpret_tensor(buf1055, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        buf1064 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch1x1_8, x_444], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_106.run(buf961, buf965, buf966, primals_171, primals_172, buf969, buf1064, 2560, 64, grid=grid(2560, 64), stream=stream0)
        del buf966
        del primals_172
        # Source Nodes: [x_448], Original ATen: [aten.convolution]
        buf970 = extern_kernels.convolution(buf959, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf970, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf971 = reinterpret_tensor(buf934, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf934  # reuse
        # Source Nodes: [x_448], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf970, buf971, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf972 = buf938; del buf938  # reuse
        buf973 = buf937; del buf937  # reuse
        buf974 = buf936; del buf936  # reuse
        # Source Nodes: [x_449], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf971, buf972, buf973, buf974, 1536, 128, grid=grid(1536), stream=stream0)
        buf975 = buf940; del buf940  # reuse
        buf976 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf978 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_449], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf972, buf973, buf974, primals_544, primals_545, buf975, buf976, buf978, primals_544, primals_545, 384, 4, grid=grid(384), stream=stream0)
        del primals_544
        del primals_545
        buf979 = reinterpret_tensor(buf970, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf970  # reuse
        # Source Nodes: [branch3x3_5, x_449], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_110.run(buf971, buf975, buf976, primals_173, primals_174, buf979, 196608, grid=grid(196608), stream=stream0)
        del primals_174
        # Source Nodes: [x_453], Original ATen: [aten.convolution]
        buf980 = extern_kernels.convolution(buf979, buf49, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf980, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf981 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_453], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf980, buf981, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf982 = buf974; del buf974  # reuse
        buf983 = buf973; del buf973  # reuse
        buf984 = buf972; del buf972  # reuse
        # Source Nodes: [x_454], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf981, buf982, buf983, buf984, 1536, 128, grid=grid(1536), stream=stream0)
        buf985 = buf976; del buf976  # reuse
        buf986 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf988 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf982, buf983, buf984, primals_547, primals_548, buf985, buf986, buf988, primals_547, primals_548, 384, 4, grid=grid(384), stream=stream0)
        del primals_547
        del primals_548
        buf1000 = buf944; del buf944  # reuse
        buf989 = reinterpret_tensor(buf1000, (8, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1063 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_454, x_457], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf981, buf985, buf986, primals_175, primals_176, buf989, buf1063, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_176
        # Source Nodes: [x_458], Original ATen: [aten.convolution]
        buf990 = extern_kernels.convolution(buf979, buf50, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf990, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf991 = reinterpret_tensor(buf980, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf980  # reuse
        # Source Nodes: [x_458], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf990, buf991, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf992 = buf984; del buf984  # reuse
        buf993 = buf983; del buf983  # reuse
        buf994 = buf982; del buf982  # reuse
        # Source Nodes: [x_459], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf991, buf992, buf993, buf994, 1536, 128, grid=grid(1536), stream=stream0)
        buf995 = buf986; del buf986  # reuse
        buf996 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf998 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_459], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf992, buf993, buf994, primals_550, primals_551, buf995, buf996, buf998, primals_550, primals_551, 384, 4, grid=grid(384), stream=stream0)
        del primals_550
        del primals_551
        buf999 = reinterpret_tensor(buf1000, (8, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1062 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_459, x_462], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf991, buf995, buf996, primals_177, primals_178, buf999, buf1062, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_178
        del buf989
        del buf999
        # Source Nodes: [x_463], Original ATen: [aten.convolution]
        buf1001 = extern_kernels.convolution(buf959, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1001, (8, 448, 8, 8), (28672, 64, 8, 1))
        buf1002 = empty_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_463], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_112.run(buf1001, buf1002, 3584, 64, grid=grid(3584, 64), stream=stream0)
        buf1003 = buf908; del buf908  # reuse
        buf1004 = buf907; del buf907  # reuse
        buf1005 = buf906; del buf906  # reuse
        # Source Nodes: [x_464], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_113.run(buf1002, buf1003, buf1004, buf1005, 1792, 128, grid=grid(1792), stream=stream0)
        buf1006 = buf910; del buf910  # reuse
        buf1007 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1009 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_464], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_114.run(buf1003, buf1004, buf1005, primals_553, primals_554, buf1006, buf1007, buf1009, primals_553, primals_554, 448, 4, grid=grid(448), stream=stream0)
        del buf1003
        del buf1004
        del buf1005
        del primals_553
        del primals_554
        buf1010 = reinterpret_tensor(buf1001, (8, 448, 8, 8), (28672, 1, 3584, 448), 0); del buf1001  # reuse
        # Source Nodes: [branch3x3dbl_15, x_464], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_115.run(buf1002, buf1006, buf1007, primals_179, primals_180, buf1010, 229376, grid=grid(229376), stream=stream0)
        del buf1007
        del primals_180
        # Source Nodes: [x_468], Original ATen: [aten.convolution]
        buf1011 = extern_kernels.convolution(buf1010, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1011, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf1012 = reinterpret_tensor(buf990, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf990  # reuse
        # Source Nodes: [x_468], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf1011, buf1012, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf1013 = buf994; del buf994  # reuse
        buf1014 = buf993; del buf993  # reuse
        buf1015 = buf992; del buf992  # reuse
        # Source Nodes: [x_469], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf1012, buf1013, buf1014, buf1015, 1536, 128, grid=grid(1536), stream=stream0)
        buf1016 = buf996; del buf996  # reuse
        buf1017 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf1019 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_469], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf1013, buf1014, buf1015, primals_556, primals_557, buf1016, buf1017, buf1019, primals_556, primals_557, 384, 4, grid=grid(384), stream=stream0)
        del primals_556
        del primals_557
        buf1020 = reinterpret_tensor(buf1011, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf1011  # reuse
        # Source Nodes: [branch3x3dbl_16, x_469], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_110.run(buf1012, buf1016, buf1017, primals_181, primals_182, buf1020, 196608, grid=grid(196608), stream=stream0)
        del primals_182
        # Source Nodes: [x_473], Original ATen: [aten.convolution]
        buf1021 = extern_kernels.convolution(buf1020, buf52, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1021, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf1022 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_473], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf1021, buf1022, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf1023 = buf1015; del buf1015  # reuse
        buf1024 = buf1014; del buf1014  # reuse
        buf1025 = buf1013; del buf1013  # reuse
        # Source Nodes: [x_474], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf1022, buf1023, buf1024, buf1025, 1536, 128, grid=grid(1536), stream=stream0)
        buf1026 = buf1017; del buf1017  # reuse
        buf1027 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf1029 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_474], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf1023, buf1024, buf1025, primals_559, primals_560, buf1026, buf1027, buf1029, primals_559, primals_560, 384, 4, grid=grid(384), stream=stream0)
        del primals_559
        del primals_560
        buf1041 = buf903; del buf903  # reuse
        buf1030 = reinterpret_tensor(buf1041, (8, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1061 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_474, x_477], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf1022, buf1026, buf1027, primals_183, primals_184, buf1030, buf1061, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_184
        # Source Nodes: [x_478], Original ATen: [aten.convolution]
        buf1031 = extern_kernels.convolution(buf1020, buf53, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1031, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf1032 = reinterpret_tensor(buf1021, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf1021  # reuse
        # Source Nodes: [x_478], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf1031, buf1032, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del buf1031
        buf1033 = buf1025; del buf1025  # reuse
        buf1034 = buf1024; del buf1024  # reuse
        buf1035 = buf1023; del buf1023  # reuse
        # Source Nodes: [x_479], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf1032, buf1033, buf1034, buf1035, 1536, 128, grid=grid(1536), stream=stream0)
        buf1036 = buf1027; del buf1027  # reuse
        buf1037 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf1039 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_479], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf1033, buf1034, buf1035, primals_562, primals_563, buf1036, buf1037, buf1039, primals_562, primals_563, 384, 4, grid=grid(384), stream=stream0)
        del buf1033
        del buf1034
        del buf1035
        del primals_562
        del primals_563
        buf1040 = reinterpret_tensor(buf1041, (8, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1060 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_479, x_482], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_111.run(buf1032, buf1036, buf1037, primals_185, primals_186, buf1040, buf1060, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del buf1037
        del primals_186
        buf1042 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_120.run(buf959, buf1042, 1048576, grid=grid(1048576), stream=stream0)
        del buf1030
        del buf1040
        # Source Nodes: [x_483], Original ATen: [aten.convolution]
        buf1043 = extern_kernels.convolution(buf1042, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1043, (8, 192, 8, 8), (12288, 64, 8, 1))
        buf1044 = reinterpret_tensor(buf946, (8, 192, 8, 8), (12288, 1, 1536, 192), 0); del buf946  # reuse
        # Source Nodes: [x_483], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf1043, buf1044, 1536, 64, grid=grid(1536, 64), stream=stream0)
        del buf1043
        buf1045 = buf950; del buf950  # reuse
        buf1046 = buf949; del buf949  # reuse
        buf1047 = buf948; del buf948  # reuse
        # Source Nodes: [x_484], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf1044, buf1045, buf1046, buf1047, 768, 128, grid=grid(768), stream=stream0)
        buf1048 = buf952; del buf952  # reuse
        buf1049 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf1051 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_484], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf1045, buf1046, buf1047, primals_565, primals_566, buf1048, buf1049, buf1051, primals_565, primals_566, 192, 4, grid=grid(192), stream=stream0)
        del buf1045
        del buf1046
        del buf1047
        del primals_565
        del primals_566
        buf1052 = reinterpret_tensor(buf1055, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        buf1059 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [branch_pool_19, x_484], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_117.run(buf1044, buf1048, buf1049, primals_187, primals_188, buf1052, buf1059, 1536, 64, grid=grid(1536, 64), stream=stream0)
        del buf1049
        del primals_188
        buf1053 = reinterpret_tensor(buf1055, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_118.run(buf1000, buf1053, 393216, grid=grid(393216), stream=stream0)
        del buf1000
        buf1054 = reinterpret_tensor(buf1055, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_118.run(buf1041, buf1054, 393216, grid=grid(393216), stream=stream0)
        del buf1041
        buf1056 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf1057 = reinterpret_tensor(buf1056, (8, 2048), (2048, 1), 0); del buf1056  # reuse
        # Source Nodes: [x_491, x_493], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_121.run(buf1057, buf1055, 16384, 64, grid=grid(16384), stream=stream0)
        del buf1052
        del buf1053
        del buf1054
        del buf1055
        del buf969
        buf1058 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_284, buf1057, reinterpret_tensor(primals_283, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf1058)
        del primals_284
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_393, primals_393, 1, grid=grid(1), stream=stream0)
        del primals_393
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_396, primals_396, 1, grid=grid(1), stream=stream0)
        del primals_396
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_399, primals_399, 1, grid=grid(1), stream=stream0)
        del primals_399
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_402, primals_402, 1, grid=grid(1), stream=stream0)
        del primals_402
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_405, primals_405, 1, grid=grid(1), stream=stream0)
        del primals_405
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_408, primals_408, 1, grid=grid(1), stream=stream0)
        del primals_408
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_411, primals_411, 1, grid=grid(1), stream=stream0)
        del primals_411
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_414, primals_414, 1, grid=grid(1), stream=stream0)
        del primals_414
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_417, primals_417, 1, grid=grid(1), stream=stream0)
        del primals_417
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_420, primals_420, 1, grid=grid(1), stream=stream0)
        del primals_420
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_423, primals_423, 1, grid=grid(1), stream=stream0)
        del primals_423
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_426, primals_426, 1, grid=grid(1), stream=stream0)
        del primals_426
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_429, primals_429, 1, grid=grid(1), stream=stream0)
        del primals_429
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_432, primals_432, 1, grid=grid(1), stream=stream0)
        del primals_432
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_435, primals_435, 1, grid=grid(1), stream=stream0)
        del primals_435
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_438, primals_438, 1, grid=grid(1), stream=stream0)
        del primals_438
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_441, primals_441, 1, grid=grid(1), stream=stream0)
        del primals_441
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_444, primals_444, 1, grid=grid(1), stream=stream0)
        del primals_444
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_447, primals_447, 1, grid=grid(1), stream=stream0)
        del primals_447
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_450, primals_450, 1, grid=grid(1), stream=stream0)
        del primals_450
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_453, primals_453, 1, grid=grid(1), stream=stream0)
        del primals_453
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_456, primals_456, 1, grid=grid(1), stream=stream0)
        del primals_456
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_459, primals_459, 1, grid=grid(1), stream=stream0)
        del primals_459
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_462, primals_462, 1, grid=grid(1), stream=stream0)
        del primals_462
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_465, primals_465, 1, grid=grid(1), stream=stream0)
        del primals_465
        # Source Nodes: [add__61], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_468, primals_468, 1, grid=grid(1), stream=stream0)
        del primals_468
        # Source Nodes: [add__62], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_471, primals_471, 1, grid=grid(1), stream=stream0)
        del primals_471
        # Source Nodes: [add__63], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_474, primals_474, 1, grid=grid(1), stream=stream0)
        del primals_474
        # Source Nodes: [add__64], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_477, primals_477, 1, grid=grid(1), stream=stream0)
        del primals_477
        # Source Nodes: [add__65], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_480, primals_480, 1, grid=grid(1), stream=stream0)
        del primals_480
        # Source Nodes: [add__66], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_483, primals_483, 1, grid=grid(1), stream=stream0)
        del primals_483
        # Source Nodes: [add__67], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_486, primals_486, 1, grid=grid(1), stream=stream0)
        del primals_486
        # Source Nodes: [add__68], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_489, primals_489, 1, grid=grid(1), stream=stream0)
        del primals_489
        # Source Nodes: [add__69], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_492, primals_492, 1, grid=grid(1), stream=stream0)
        del primals_492
        # Source Nodes: [add__70], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_495, primals_495, 1, grid=grid(1), stream=stream0)
        del primals_495
        # Source Nodes: [add__71], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_498, primals_498, 1, grid=grid(1), stream=stream0)
        del primals_498
        # Source Nodes: [add__72], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_501, primals_501, 1, grid=grid(1), stream=stream0)
        del primals_501
        # Source Nodes: [add__73], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_504, primals_504, 1, grid=grid(1), stream=stream0)
        del primals_504
        # Source Nodes: [add__74], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_507, primals_507, 1, grid=grid(1), stream=stream0)
        del primals_507
        # Source Nodes: [add__75], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_510, primals_510, 1, grid=grid(1), stream=stream0)
        del primals_510
        # Source Nodes: [add__76], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_513, primals_513, 1, grid=grid(1), stream=stream0)
        del primals_513
        # Source Nodes: [add__77], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_516, primals_516, 1, grid=grid(1), stream=stream0)
        del primals_516
        # Source Nodes: [add__78], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_519, primals_519, 1, grid=grid(1), stream=stream0)
        del primals_519
        # Source Nodes: [add__79], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_522, primals_522, 1, grid=grid(1), stream=stream0)
        del primals_522
        # Source Nodes: [add__80], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_525, primals_525, 1, grid=grid(1), stream=stream0)
        del primals_525
        # Source Nodes: [add__81], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_528, primals_528, 1, grid=grid(1), stream=stream0)
        del primals_528
        # Source Nodes: [add__82], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_531, primals_531, 1, grid=grid(1), stream=stream0)
        del primals_531
        # Source Nodes: [add__83], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_534, primals_534, 1, grid=grid(1), stream=stream0)
        del primals_534
        # Source Nodes: [add__84], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_537, primals_537, 1, grid=grid(1), stream=stream0)
        del primals_537
        # Source Nodes: [add__85], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_540, primals_540, 1, grid=grid(1), stream=stream0)
        del primals_540
        # Source Nodes: [add__86], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_543, primals_543, 1, grid=grid(1), stream=stream0)
        del primals_543
        # Source Nodes: [add__87], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_546, primals_546, 1, grid=grid(1), stream=stream0)
        del primals_546
        # Source Nodes: [add__88], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_549, primals_549, 1, grid=grid(1), stream=stream0)
        del primals_549
        # Source Nodes: [add__89], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_552, primals_552, 1, grid=grid(1), stream=stream0)
        del primals_552
        # Source Nodes: [add__90], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_555, primals_555, 1, grid=grid(1), stream=stream0)
        del primals_555
        # Source Nodes: [add__91], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_558, primals_558, 1, grid=grid(1), stream=stream0)
        del primals_558
        # Source Nodes: [add__92], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_561, primals_561, 1, grid=grid(1), stream=stream0)
        del primals_561
        # Source Nodes: [add__93], Original ATen: [aten.add]
        triton_poi_fused_add_122.run(primals_564, primals_564, 1, grid=grid(1), stream=stream0)
        del primals_564
        return (buf1058, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, buf0, buf1, buf2, primals_192, buf3, primals_194, primals_195, buf4, primals_197, buf5, buf6, primals_200, primals_201, primals_202, buf7, primals_204, buf8, buf9, primals_207, primals_208, primals_209, buf10, primals_211, buf11, buf12, primals_214, buf13, primals_216, buf14, buf15, primals_219, primals_220, buf16, buf17, primals_223, buf18, buf19, buf20, buf21, primals_228, primals_229, primals_230, buf22, buf23, primals_233, buf24, buf25, buf26, buf27, primals_238, primals_239, primals_240, buf28, buf29, primals_243, buf30, buf31, buf32, buf33, primals_248, primals_249, primals_250, buf34, buf35, primals_253, buf36, buf37, buf38, buf39, primals_258, primals_259, buf40, primals_261, buf41, buf42, buf43, primals_265, primals_266, buf44, buf45, primals_269, buf46, buf47, buf48, primals_273, primals_274, primals_275, buf49, buf50, primals_278, buf51, buf52, buf53, primals_282, buf54, buf56, buf66, buf67, buf69, buf79, buf80, buf82, buf92, buf93, buf94, buf95, buf97, buf107, buf108, buf110, buf120, buf121, buf122, buf123, buf125, buf132, buf135, buf142, buf143, buf145, buf152, buf155, buf162, buf163, buf165, buf172, buf173, buf175, buf182, buf184, buf186, buf193, buf196, buf198, buf205, buf208, buf215, buf216, buf218, buf225, buf228, buf235, buf236, buf238, buf245, buf246, buf248, buf255, buf257, buf259, buf266, buf269, buf271, buf278, buf281, buf288, buf289, buf291, buf298, buf301, buf308, buf309, buf311, buf318, buf319, buf321, buf328, buf330, buf332, buf339, buf342, buf344, buf351, buf354, buf361, buf362, buf364, buf371, buf372, buf374, buf381, buf384, buf386, buf388, buf395, buf398, buf405, buf406, buf408, buf415, buf416, buf418, buf425, buf428, buf435, buf436, buf438, buf445, buf446, buf448, buf455, buf456, buf458, buf465, buf466, buf468, buf475, buf477, buf479, buf486, buf489, buf491, buf498, buf501, buf508, buf509, buf511, buf518, buf519, buf521, buf528, buf531, buf538, buf539, buf541, buf548, buf549, buf551, buf558, buf559, buf561, buf568, buf569, buf571, buf578, buf580, buf582, buf589, buf592, buf594, buf601, buf604, buf611, buf612, buf614, buf621, buf622, buf624, buf631, buf634, buf641, buf642, buf644, buf651, buf652, buf654, buf661, buf662, buf664, buf671, buf672, buf674, buf681, buf683, buf685, buf692, buf695, buf697, buf704, buf707, buf714, buf715, buf717, buf724, buf725, buf727, buf734, buf737, buf744, buf745, buf747, buf754, buf755, buf757, buf764, buf765, buf767, buf774, buf775, buf777, buf784, buf786, buf788, buf795, buf798, buf800, buf807, buf808, buf810, buf817, buf820, buf827, buf828, buf830, buf837, buf838, buf840, buf847, buf848, buf850, buf857, buf860, buf862, buf864, buf871, buf874, buf881, buf882, buf884, buf891, buf894, buf901, buf905, buf912, buf913, buf915, buf922, buf923, buf925, buf932, buf935, buf942, buf945, buf947, buf954, buf959, buf961, buf968, buf971, buf978, buf979, buf981, buf988, buf991, buf998, buf1002, buf1009, buf1010, buf1012, buf1019, buf1020, buf1022, buf1029, buf1032, buf1039, buf1042, buf1044, buf1051, buf1057, reinterpret_tensor(primals_283, (1000, 2048), (2048, 1), 0), buf1059, reinterpret_tensor(buf1048, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1060, reinterpret_tensor(buf1036, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1061, reinterpret_tensor(buf1026, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf1016, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf1006, (1, 448, 1, 1), (448, 1, 1, 1), 0), buf1062, reinterpret_tensor(buf995, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1063, reinterpret_tensor(buf985, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf975, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1064, reinterpret_tensor(buf965, (1, 320, 1, 1), (320, 1, 1, 1), 0), buf1065, reinterpret_tensor(buf951, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1066, reinterpret_tensor(buf939, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1067, reinterpret_tensor(buf929, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf919, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf909, (1, 448, 1, 1), (448, 1, 1, 1), 0), buf1068, reinterpret_tensor(buf898, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1069, reinterpret_tensor(buf888, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf878, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1070, reinterpret_tensor(buf868, (1, 320, 1, 1), (320, 1, 1, 1), 0), buf1071, reinterpret_tensor(buf854, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf844, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf834, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf824, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1072, reinterpret_tensor(buf814, (1, 320, 1, 1), (320, 1, 1, 1), 0), reinterpret_tensor(buf804, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1073, reinterpret_tensor(buf792, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1074, reinterpret_tensor(buf781, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf771, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf761, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf751, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf741, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1075, reinterpret_tensor(buf731, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf721, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf711, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1076, reinterpret_tensor(buf701, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1077, reinterpret_tensor(buf689, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1078, reinterpret_tensor(buf678, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf668, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf658, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf648, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf638, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1079, reinterpret_tensor(buf628, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf618, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf608, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1080, reinterpret_tensor(buf598, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1081, reinterpret_tensor(buf586, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1082, reinterpret_tensor(buf575, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf565, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf555, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf545, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf535, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1083, reinterpret_tensor(buf525, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf515, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf505, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1084, reinterpret_tensor(buf495, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1085, reinterpret_tensor(buf483, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1086, reinterpret_tensor(buf472, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf462, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf452, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf442, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf432, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1087, reinterpret_tensor(buf422, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf412, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf402, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1088, reinterpret_tensor(buf392, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1089, reinterpret_tensor(buf378, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf368, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf358, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1090, reinterpret_tensor(buf348, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1091, reinterpret_tensor(buf336, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1092, reinterpret_tensor(buf325, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf305, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1093, reinterpret_tensor(buf295, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1094, reinterpret_tensor(buf275, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1095, reinterpret_tensor(buf263, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1096, reinterpret_tensor(buf252, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf242, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf232, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1097, reinterpret_tensor(buf222, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf212, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1098, reinterpret_tensor(buf202, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1099, reinterpret_tensor(buf190, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf1100, reinterpret_tensor(buf179, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf169, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf159, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1101, reinterpret_tensor(buf149, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf139, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1102, reinterpret_tensor(buf129, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf117, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf104, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf89, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf76, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_370 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_373 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_385 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_400 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_403 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_406 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_409 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_412 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_415 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_418 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_421 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_424 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_427 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_430 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_433 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_436 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_439 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_442 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_445 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_448 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_451 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_454 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_457 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_460 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_463 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_466 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_469 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_472 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_475 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_478 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_481 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_484 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_487 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_490 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_493 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_496 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_499 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_502 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_505 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_508 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_511 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_514 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_517 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_520 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_523 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_526 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_529 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_532 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_535 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_538 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_541 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_544 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_547 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_550 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_553 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_556 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_559 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_562 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_565 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((8, 3, 299, 299), (268203, 89401, 299, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('inception_v3', benchmark_compiled_module)
