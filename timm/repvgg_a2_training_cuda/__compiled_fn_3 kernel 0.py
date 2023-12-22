
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


# kernel path: /tmp/torchinductor_youkaichao/xy/cxygl4r7pth2qydfjrtx5s76nk436z2r62o3vxgknrl7lhquc65y.py
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
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zs/czsgn3gdvxxth2bkbcfg7a3j4mebzix6zmabrg5qhq5pstxre5nf.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmzglsl4lcmicb4l35qmmcuvfjge7rjup57z3r3mbofe6rmu2zz.py
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
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36oiagljczxxm76j6yz72kzdoewp76jeazhnqqoyws5htz65rnr.py
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


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jtrmqumaousb2c5bodsrtukrvhookkd44haeoef46dhrlxipgw.py
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
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/eh/cehmnoqi4rrfoc45yki6kcvdsojdiw42wqw75qeiheuch7x3nqbr.py
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
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
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


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcg6j6cl23lk2vils26c5wutgdc4cmohez7wmw6h5m5harisw2i.py
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
    size_hints=[1048576, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 540672
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


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pi2ijarbajswgtyofnmg75pon54ymfcpwvv25o34syvi7tdz6c.py
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
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yb/cybbgkyhbdgdxavsdxekdwyj4hlemtymyjvpk6ro4xhfgdbskqaf.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgeprlj22hjmg6vmmsfmtfkwczkog5f7vt4v4wjvrgfiysnj3nua.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qo/cqoe6ltbju3zn6cenuz7pxoxxsbcvzoxqytagapd54jnwqavdb6x.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lo/clofujfsjmmvn6jqud47c5ya6mks7agkgl6pp6uyqzzlrrmnxyaa.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcet4pdg73m727d5drzdwmqrupqswrhv7zskwykqs3tfnqa2wig.py
# Source Nodes: [x_1, x_10, x_12, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_10 => add_10
# x_12 => relu
# x_6 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_add_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2zacshko5jv5i53ujddujbxkzgfpm3sumcz5pwrkywoe5zke3v.py
# Source Nodes: [x_13], Original ATen: [aten.convolution]
# x_13 => convolution_2
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5x3xcsjl5y7mnytfmcqxyzipntwvgzlh7kcvz5cls4ie5glrp2f.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bo/cbogd75rh3hezdqlkbvjzjre53w3n5ezlygckoixjznf7svqk35r.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (96*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (96*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (96*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bqx632znkt63rs7e7wocqn6aj3xg42thytzw6jeit4oee5ej34.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
# x_14 => add_12, add_13, add_14, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/hk/chkpvwd6iqmbixjtlwjqbdcjn67f47deb42mjpeh6prgudmcfshl.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_14, x_19, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___0_____0___act => relu_1
# x_14 => add_12, add_15, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_19 => add_17, add_20, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_23 => add_21
triton_poi_fused__native_batch_norm_legit_functional_add_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a3f4bcdvh6vlxwljbsqwdsnr3pityqykopccbt6lbj5motyiin.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____1___act, x_25, x_29, x_34, x_38, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___0_____1___act => relu_2
# x_25 => add_23, add_26, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_29 => add_28, add_31, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_34 => add_33, add_36, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_38 => add_37
# x_40 => add_38
triton_poi_fused__native_batch_norm_legit_functional_add_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr10 + (x2), None)
    tmp28 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 / tmp4
    tmp32 = tmp31 + tmp6
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp29 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp26 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tl.store(in_out_ptr0 + (x2), tmp40, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vdkjm32q6nh3agdnm2qqjg26krjzku75enjawzfb5ibnveen4p.py
# Source Nodes: [x_42], Original ATen: [aten.convolution]
# x_42 => convolution_6
triton_poi_fused_convolution_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjkwwhp3ae3uvbuwnlz4xinyfyzc6ctijakysr4yghofd7dw2l2.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
# x_43 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
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


# kernel path: /tmp/torchinductor_youkaichao/en/cen6iz4xjwrorela5pmbi3fu3qfyahx2rnjxecxmz57764qm67px.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
# x_43 => add_40, add_41, add_42, mul_50, mul_51, mul_52, mul_53, mul_54, rsqrt_7, squeeze_22, var_mean_7
triton_per_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001594642002871
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


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zcjoaqwdem74pyppok2zaogrhcyibeuomg3vdtmo4iuppakgxg.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_43, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___1_____0___act => relu_3
# x_43 => add_40, add_43, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_48 => add_45, add_48, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
# x_52 => add_49
triton_poi_fused__native_batch_norm_legit_functional_add_relu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5przoc5nfy66ryikiomizg2f3my4xw5yscnp4r6bteagejj5ho.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act, x_54, x_58, x_63, x_67, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___1_____1___act => relu_4
# x_54 => add_51, add_54, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# x_58 => add_56, add_59, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# x_63 => add_61, add_64, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# x_67 => add_65
# x_69 => add_66
triton_poi_fused__native_batch_norm_legit_functional_add_relu_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr10 + (x2), None)
    tmp28 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 / tmp4
    tmp32 = tmp31 + tmp6
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp29 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp26 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tl.store(in_out_ptr0 + (x2), tmp40, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxhdtqjr64lfyx4ngjs4lwmvthrwjrkktgs77dkyso2scrb5xbs.py
# Source Nodes: [x_105], Original ATen: [aten.convolution]
# x_105 => convolution_14
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscqgspsds6vuotec64lqn6bgvbzwlu46bfeffmnh3o3g5isvm34.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcmmdmgxyv3yhorxchffgjxxuyewif2vjyxcxqx2phctyyraqkw.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => add_102, add_103, add_104, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6txy7kly5lfacdhcgzlsfxy3gxkmbsde3j3ne5w6ud5epdggqto.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_106, x_111, x_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___2_____0___act => relu_7
# x_106 => add_102, add_105, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# x_111 => add_107, add_110, mul_133, mul_139, rsqrt_19, sub_19, var_mean_19
# x_115 => add_111
triton_poi_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyvoyiay5kmwve2yegjbnd5vecg34vulr5iam5rys2wul4jsboq.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act, x_117, x_121, x_126, x_130, x_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___2_____1___act => relu_8
# x_117 => add_113, add_116, mul_140, mul_146, rsqrt_20, sub_20, var_mean_20
# x_121 => add_118, add_121, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# x_126 => add_123, add_126, mul_154, mul_160, rsqrt_22, sub_22, var_mean_22
# x_130 => add_127
# x_132 => add_128
triton_poi_fused__native_batch_norm_legit_functional_add_relu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr10 + (x2), None)
    tmp28 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 / tmp4
    tmp32 = tmp31 + tmp6
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp29 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp26 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tl.store(in_out_ptr0 + (x2), tmp40, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfijekhdnn5ezijgvuvlrngvrnerpx4ox2ccrjbosyjgtb6wms6b.py
# Source Nodes: [x_338], Original ATen: [aten.convolution]
# x_338 => convolution_42
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 11264
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1408
    y1 = (yindex // 1408)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1408*x2) + (68992*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4tlj7tcieyrfvapf2vsxmssg4ts6ncuedh3mtwqigxejk7qj3g.py
# Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
# x_339 => var_mean_59
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5632
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1408
    x1 = (xindex // 1408)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1408*r2) + (137984*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7p/c7prm6bwbmhornat4t4byoveji2ggi4me7oxliy3c3vuy3duxjdm.py
# Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
# x_339 => add_334, add_335, add_336, mul_414, mul_415, mul_416, mul_417, mul_418, rsqrt_59, squeeze_178, var_mean_59
triton_per_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1408
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1408*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1408*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1408*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/sn/csnt543qy2rzph2pyk56jxirdr5aecmwvim5i63quwgj2xty3q3x.py
# Source Nodes: [x_339, x_344, x_348, x_350], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# x_339 => add_334, add_337, mul_413, mul_419, rsqrt_59, sub_59, var_mean_59
# x_344 => add_339, add_342, mul_420, mul_426, rsqrt_60, sub_60, var_mean_60
# x_348 => add_343
# x_350 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*i1', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 551936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1408
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp15 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tmp28 = 0.0
    tmp29 = tmp27 <= tmp28
    tl.store(out_ptr0 + (x2), tmp26, xmask)
    tl.store(out_ptr1 + (x2), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c554qzcvckfp3mygbdhohyy7v35n576bpw67i3yss2qkpo6n4ydt.py
# Source Nodes: [x_350, x_353, x_355], Original ATen: [aten.mean, aten.relu, aten.view]
# x_350 => relu_21
# x_353 => mean
# x_355 => view
triton_per_fused_mean_relu_view_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_relu_view_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1408
    x1 = (xindex // 1408)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1408*r2) + (68992*x1)), rmask & xmask, other=0.0)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 49.0
    tmp7 = tmp5 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuexgdxtrk5f5y3m2xm7r2bhxqivg44dbc7lk2opwy7beijier5.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_35', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (96, ), (1, ))
    assert_size_stride(primals_6, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_14, (96, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (192, ), (1, ))
    assert_size_stride(primals_19, (192, ), (1, ))
    assert_size_stride(primals_20, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (192, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (384, ), (1, ))
    assert_size_stride(primals_60, (384, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_98, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_104, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (384, ), (1, ))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (384, ), (1, ))
    assert_size_stride(primals_116, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (1408, ), (1, ))
    assert_size_stride(primals_120, (1408, ), (1, ))
    assert_size_stride(primals_121, (1408, ), (1, ))
    assert_size_stride(primals_122, (1408, ), (1, ))
    assert_size_stride(primals_123, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_124, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_125, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_126, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_127, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_128, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_129, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_130, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_131, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_132, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_133, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_134, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_135, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_136, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_137, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_138, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_139, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_140, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_141, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_142, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_143, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_144, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_145, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_146, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_147, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_148, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_149, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_150, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_151, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_152, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_153, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_154, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_155, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_156, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_157, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_158, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_159, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_160, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_161, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_162, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_163, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_164, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_165, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_166, (1408, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_167, (1000, 1408), (1408, 1))
    assert_size_stride(primals_168, (1000, ), (1, ))
    assert_size_stride(primals_169, (), ())
    assert_size_stride(primals_170, (64, ), (1, ))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (), ())
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (), ())
    assert_size_stride(primals_176, (96, ), (1, ))
    assert_size_stride(primals_177, (96, ), (1, ))
    assert_size_stride(primals_178, (), ())
    assert_size_stride(primals_179, (96, ), (1, ))
    assert_size_stride(primals_180, (96, ), (1, ))
    assert_size_stride(primals_181, (), ())
    assert_size_stride(primals_182, (96, ), (1, ))
    assert_size_stride(primals_183, (96, ), (1, ))
    assert_size_stride(primals_184, (), ())
    assert_size_stride(primals_185, (96, ), (1, ))
    assert_size_stride(primals_186, (96, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (96, ), (1, ))
    assert_size_stride(primals_189, (96, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (192, ), (1, ))
    assert_size_stride(primals_192, (192, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_195, (192, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_198, (192, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (192, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (192, ), (1, ))
    assert_size_stride(primals_208, (), ())
    assert_size_stride(primals_209, (192, ), (1, ))
    assert_size_stride(primals_210, (192, ), (1, ))
    assert_size_stride(primals_211, (), ())
    assert_size_stride(primals_212, (192, ), (1, ))
    assert_size_stride(primals_213, (192, ), (1, ))
    assert_size_stride(primals_214, (), ())
    assert_size_stride(primals_215, (192, ), (1, ))
    assert_size_stride(primals_216, (192, ), (1, ))
    assert_size_stride(primals_217, (), ())
    assert_size_stride(primals_218, (192, ), (1, ))
    assert_size_stride(primals_219, (192, ), (1, ))
    assert_size_stride(primals_220, (), ())
    assert_size_stride(primals_221, (192, ), (1, ))
    assert_size_stride(primals_222, (192, ), (1, ))
    assert_size_stride(primals_223, (), ())
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (), ())
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_229, (), ())
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (), ())
    assert_size_stride(primals_233, (384, ), (1, ))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (), ())
    assert_size_stride(primals_236, (384, ), (1, ))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (), ())
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (), ())
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (), ())
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_247, (), ())
    assert_size_stride(primals_248, (384, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (), ())
    assert_size_stride(primals_251, (384, ), (1, ))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (384, ), (1, ))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (384, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (384, ), (1, ))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (384, ), (1, ))
    assert_size_stride(primals_270, (384, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (384, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (384, ), (1, ))
    assert_size_stride(primals_276, (384, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (384, ), (1, ))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (384, ), (1, ))
    assert_size_stride(primals_282, (384, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (384, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (384, ), (1, ))
    assert_size_stride(primals_288, (384, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (384, ), (1, ))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (384, ), (1, ))
    assert_size_stride(primals_294, (384, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (384, ), (1, ))
    assert_size_stride(primals_297, (384, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (384, ), (1, ))
    assert_size_stride(primals_300, (384, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (384, ), (1, ))
    assert_size_stride(primals_303, (384, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (384, ), (1, ))
    assert_size_stride(primals_306, (384, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (384, ), (1, ))
    assert_size_stride(primals_309, (384, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (384, ), (1, ))
    assert_size_stride(primals_312, (384, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_315, (384, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (384, ), (1, ))
    assert_size_stride(primals_318, (384, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (384, ), (1, ))
    assert_size_stride(primals_321, (384, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (384, ), (1, ))
    assert_size_stride(primals_324, (384, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (384, ), (1, ))
    assert_size_stride(primals_327, (384, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (384, ), (1, ))
    assert_size_stride(primals_330, (384, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (384, ), (1, ))
    assert_size_stride(primals_333, (384, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (384, ), (1, ))
    assert_size_stride(primals_336, (384, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (384, ), (1, ))
    assert_size_stride(primals_339, (384, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (384, ), (1, ))
    assert_size_stride(primals_342, (384, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (384, ), (1, ))
    assert_size_stride(primals_345, (384, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (1408, ), (1, ))
    assert_size_stride(primals_348, (1408, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (1408, ), (1, ))
    assert_size_stride(primals_351, (1408, ), (1, ))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_124, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_124
        buf1 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_126, buf1, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_126
        buf2 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_128, buf2, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_128
        buf3 = empty_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_130, buf3, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del primals_130
        buf4 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_132, buf4, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_132
        buf5 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_134, buf5, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_134
        buf6 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_136, buf6, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_136
        buf7 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_138, buf7, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_138
        buf8 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_140, buf8, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_140
        buf9 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_142, buf9, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_142
        buf10 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_144, buf10, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_144
        buf11 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_146, buf11, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_146
        buf12 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_148, buf12, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_148
        buf13 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_150, buf13, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_150
        buf14 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_152, buf14, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_152
        buf15 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_154, buf15, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_154
        buf16 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_156, buf16, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_156
        buf17 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_158, buf17, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_158
        buf18 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_160, buf18, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_160
        buf19 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_162, buf19, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_162
        buf20 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_164, buf20, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_164
        buf21 = empty_strided((1408, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_166, buf21, 540672, 9, grid=grid(540672, 9), stream=stream0)
        del primals_166
        buf22 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_352, buf22, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_352
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_123, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf24 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf23, buf24, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf25 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf27 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf24, buf25, buf26, buf27, 50176, 128, grid=grid(50176), stream=stream0)
        buf28 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf25, buf26, buf27, buf28, buf29, buf30, 448, 112, grid=grid(448), stream=stream0)
        buf31 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf34 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf28, buf29, buf30, primals_170, primals_171, buf31, buf32, buf34, primals_170, primals_171, 64, 7, grid=grid(64), stream=stream0)
        del primals_170
        del primals_171
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf22, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf36 = reinterpret_tensor(buf23, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf23  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf35, buf36, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf37 = buf27; del buf27  # reuse
        buf38 = buf26; del buf26  # reuse
        buf39 = buf25; del buf25  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf36, buf37, buf38, buf39, 50176, 128, grid=grid(50176), stream=stream0)
        buf40 = buf30; del buf30  # reuse
        buf41 = buf29; del buf29  # reuse
        buf42 = buf28; del buf28  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf37, buf38, buf39, buf40, buf41, buf42, 448, 112, grid=grid(448), stream=stream0)
        del buf37
        del buf38
        del buf39
        buf43 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf46 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf40, buf41, buf42, primals_173, primals_174, buf43, buf44, buf46, primals_173, primals_174, 64, 7, grid=grid(64), stream=stream0)
        del buf40
        del buf41
        del buf42
        del primals_173
        del primals_174
        buf47 = reinterpret_tensor(buf35, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf35  # reuse
        buf48 = buf47; del buf47  # reuse
        # Source Nodes: [x_1, x_10, x_12, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_13.run(buf48, buf24, buf31, buf32, primals_1, primals_2, buf36, buf43, buf44, primals_3, primals_4, 6422528, grid=grid(6422528), stream=stream0)
        del buf32
        del buf44
        del primals_2
        del primals_4
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_125, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf50 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf49, buf50, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf51 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf50, buf51, buf52, buf53, 18816, 128, grid=grid(18816), stream=stream0)
        buf54 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf51, buf52, buf53, buf54, buf55, buf56, 192, 98, grid=grid(192), stream=stream0)
        buf57 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf60 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf54, buf55, buf56, primals_176, primals_177, buf57, buf58, buf60, primals_176, primals_177, 96, 2, grid=grid(96), stream=stream0)
        del primals_176
        del primals_177
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf48, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf62 = reinterpret_tensor(buf49, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf49  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf61, buf62, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf63 = buf53; del buf53  # reuse
        buf64 = buf52; del buf52  # reuse
        buf65 = buf51; del buf51  # reuse
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf62, buf63, buf64, buf65, 18816, 128, grid=grid(18816), stream=stream0)
        buf66 = buf56; del buf56  # reuse
        buf67 = buf55; del buf55  # reuse
        buf68 = buf54; del buf54  # reuse
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf63, buf64, buf65, buf66, buf67, buf68, 192, 98, grid=grid(192), stream=stream0)
        buf69 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf72 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf66, buf67, buf68, primals_179, primals_180, buf69, buf70, buf72, primals_179, primals_180, 96, 2, grid=grid(96), stream=stream0)
        del primals_179
        del primals_180
        buf73 = reinterpret_tensor(buf61, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf61  # reuse
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_14, x_19, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_18.run(buf74, buf50, buf57, buf58, primals_5, primals_6, buf62, buf69, buf70, primals_7, primals_8, 2408448, grid=grid(2408448), stream=stream0)
        del primals_6
        del primals_8
        buf75 = buf65; del buf65  # reuse
        buf76 = buf64; del buf64  # reuse
        buf77 = buf63; del buf63  # reuse
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf74, buf75, buf76, buf77, 18816, 128, grid=grid(18816), stream=stream0)
        buf78 = buf68; del buf68  # reuse
        buf79 = buf67; del buf67  # reuse
        buf80 = buf66; del buf66  # reuse
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf75, buf76, buf77, buf78, buf79, buf80, 192, 98, grid=grid(192), stream=stream0)
        buf81 = buf70; del buf70  # reuse
        buf82 = buf58; del buf58  # reuse
        buf84 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf78, buf79, buf80, primals_182, primals_183, buf81, buf82, buf84, primals_182, primals_183, 96, 2, grid=grid(96), stream=stream0)
        del primals_182
        del primals_183
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf74, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf86 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf85, buf86, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf87 = buf77; del buf77  # reuse
        buf88 = buf76; del buf76  # reuse
        buf89 = buf75; del buf75  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf86, buf87, buf88, buf89, 18816, 128, grid=grid(18816), stream=stream0)
        buf90 = buf80; del buf80  # reuse
        buf91 = buf79; del buf79  # reuse
        buf92 = buf78; del buf78  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf87, buf88, buf89, buf90, buf91, buf92, 192, 98, grid=grid(192), stream=stream0)
        buf93 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf96 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf90, buf91, buf92, primals_185, primals_186, buf93, buf94, buf96, primals_185, primals_186, 96, 2, grid=grid(96), stream=stream0)
        del primals_185
        del primals_186
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf74, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf98 = reinterpret_tensor(buf85, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf85  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf97, buf98, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf99 = buf89; del buf89  # reuse
        buf100 = buf88; del buf88  # reuse
        buf101 = buf87; del buf87  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf98, buf99, buf100, buf101, 18816, 128, grid=grid(18816), stream=stream0)
        buf102 = buf92; del buf92  # reuse
        buf103 = buf91; del buf91  # reuse
        buf104 = buf90; del buf90  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf99, buf100, buf101, buf102, buf103, buf104, 192, 98, grid=grid(192), stream=stream0)
        del buf100
        del buf101
        del buf99
        buf105 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf108 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf102, buf103, buf104, primals_188, primals_189, buf105, buf106, buf108, primals_188, primals_189, 96, 2, grid=grid(96), stream=stream0)
        del primals_188
        del primals_189
        buf109 = reinterpret_tensor(buf97, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf97  # reuse
        buf110 = buf109; del buf109  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____1___act, x_25, x_29, x_34, x_38, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_19.run(buf110, buf86, buf93, buf94, primals_11, primals_12, buf98, buf105, buf106, primals_13, primals_14, buf74, buf81, buf82, primals_9, primals_10, 2408448, grid=grid(2408448), stream=stream0)
        del buf106
        del buf82
        del buf94
        del primals_10
        del primals_12
        del primals_14
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_129, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf112 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf111, buf112, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf113 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf114 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf115 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf112, buf113, buf114, buf115, 9408, 128, grid=grid(9408), stream=stream0)
        buf116 = reinterpret_tensor(buf104, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf104  # reuse
        buf117 = reinterpret_tensor(buf103, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf103  # reuse
        buf119 = reinterpret_tensor(buf102, (192, ), (1, ), 0); del buf102  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf113, buf114, buf115, primals_191, primals_192, buf116, buf117, buf119, primals_191, primals_192, 192, 49, grid=grid(192), stream=stream0)
        del primals_191
        del primals_192
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf110, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf121 = reinterpret_tensor(buf111, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf111  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf120, buf121, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf122 = buf115; del buf115  # reuse
        buf123 = buf114; del buf114  # reuse
        buf124 = buf113; del buf113  # reuse
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf121, buf122, buf123, buf124, 9408, 128, grid=grid(9408), stream=stream0)
        buf125 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf126 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf128 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf122, buf123, buf124, primals_194, primals_195, buf125, buf126, buf128, primals_194, primals_195, 192, 49, grid=grid(192), stream=stream0)
        del primals_194
        del primals_195
        buf129 = reinterpret_tensor(buf120, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf120  # reuse
        buf130 = buf129; del buf129  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_43, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_23.run(buf130, buf112, buf116, buf117, primals_15, primals_16, buf121, buf125, buf126, primals_17, primals_18, 1204224, grid=grid(1204224), stream=stream0)
        del primals_16
        del primals_18
        buf131 = buf124; del buf124  # reuse
        buf132 = buf123; del buf123  # reuse
        buf133 = buf122; del buf122  # reuse
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf130, buf131, buf132, buf133, 9408, 128, grid=grid(9408), stream=stream0)
        buf134 = buf126; del buf126  # reuse
        buf135 = buf117; del buf117  # reuse
        buf137 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf131, buf132, buf133, primals_197, primals_198, buf134, buf135, buf137, primals_197, primals_198, 192, 49, grid=grid(192), stream=stream0)
        del primals_197
        del primals_198
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf130, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf139 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf138, buf139, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf140 = buf133; del buf133  # reuse
        buf141 = buf132; del buf132  # reuse
        buf142 = buf131; del buf131  # reuse
        # Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf139, buf140, buf141, buf142, 9408, 128, grid=grid(9408), stream=stream0)
        buf143 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf144 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf146 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf140, buf141, buf142, primals_200, primals_201, buf143, buf144, buf146, primals_200, primals_201, 192, 49, grid=grid(192), stream=stream0)
        del primals_200
        del primals_201
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf130, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf148 = reinterpret_tensor(buf138, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf138  # reuse
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf147, buf148, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf149 = buf142; del buf142  # reuse
        buf150 = buf141; del buf141  # reuse
        buf151 = buf140; del buf140  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf148, buf149, buf150, buf151, 9408, 128, grid=grid(9408), stream=stream0)
        buf152 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf153 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf155 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf149, buf150, buf151, primals_203, primals_204, buf152, buf153, buf155, primals_203, primals_204, 192, 49, grid=grid(192), stream=stream0)
        del primals_203
        del primals_204
        buf156 = reinterpret_tensor(buf147, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf147  # reuse
        buf157 = buf156; del buf156  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act, x_54, x_58, x_63, x_67, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_24.run(buf157, buf139, buf143, buf144, primals_21, primals_22, buf148, buf152, buf153, primals_23, primals_24, buf130, buf134, buf135, primals_19, primals_20, 1204224, grid=grid(1204224), stream=stream0)
        del primals_20
        del primals_22
        del primals_24
        buf158 = buf151; del buf151  # reuse
        buf159 = buf150; del buf150  # reuse
        buf160 = buf149; del buf149  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf157, buf158, buf159, buf160, 9408, 128, grid=grid(9408), stream=stream0)
        buf161 = buf153; del buf153  # reuse
        buf162 = buf144; del buf144  # reuse
        buf164 = reinterpret_tensor(buf135, (192, ), (1, ), 0); del buf135  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf158, buf159, buf160, primals_206, primals_207, buf161, buf162, buf164, primals_206, primals_207, 192, 49, grid=grid(192), stream=stream0)
        del primals_206
        del primals_207
        # Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf157, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf166 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf165, buf166, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf167 = buf160; del buf160  # reuse
        buf168 = buf159; del buf159  # reuse
        buf169 = buf158; del buf158  # reuse
        # Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf166, buf167, buf168, buf169, 9408, 128, grid=grid(9408), stream=stream0)
        buf170 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf173 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf167, buf168, buf169, primals_209, primals_210, buf170, buf171, buf173, primals_209, primals_210, 192, 49, grid=grid(192), stream=stream0)
        del primals_209
        del primals_210
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf157, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf175 = reinterpret_tensor(buf165, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf165  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf174, buf175, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf176 = buf169; del buf169  # reuse
        buf177 = buf168; del buf168  # reuse
        buf178 = buf167; del buf167  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf175, buf176, buf177, buf178, 9408, 128, grid=grid(9408), stream=stream0)
        buf179 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf182 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf176, buf177, buf178, primals_212, primals_213, buf179, buf180, buf182, primals_212, primals_213, 192, 49, grid=grid(192), stream=stream0)
        del primals_212
        del primals_213
        buf183 = reinterpret_tensor(buf174, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf174  # reuse
        buf184 = buf183; del buf183  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____2___act, x_71, x_75, x_80, x_84, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_24.run(buf184, buf166, buf170, buf171, primals_27, primals_28, buf175, buf179, buf180, primals_29, primals_30, buf157, buf161, buf162, primals_25, primals_26, 1204224, grid=grid(1204224), stream=stream0)
        del primals_26
        del primals_28
        del primals_30
        buf185 = buf178; del buf178  # reuse
        buf186 = buf177; del buf177  # reuse
        buf187 = buf176; del buf176  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf184, buf185, buf186, buf187, 9408, 128, grid=grid(9408), stream=stream0)
        buf188 = buf180; del buf180  # reuse
        buf189 = buf171; del buf171  # reuse
        buf191 = reinterpret_tensor(buf162, (192, ), (1, ), 0); del buf162  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf185, buf186, buf187, primals_215, primals_216, buf188, buf189, buf191, primals_215, primals_216, 192, 49, grid=grid(192), stream=stream0)
        del primals_215
        del primals_216
        # Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf184, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf193 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf192, buf193, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf194 = buf187; del buf187  # reuse
        buf195 = buf186; del buf186  # reuse
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [x_92], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf193, buf194, buf195, buf196, 9408, 128, grid=grid(9408), stream=stream0)
        buf197 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf200 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf194, buf195, buf196, primals_218, primals_219, buf197, buf198, buf200, primals_218, primals_219, 192, 49, grid=grid(192), stream=stream0)
        del primals_218
        del primals_219
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf184, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf202 = reinterpret_tensor(buf192, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf192  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf201, buf202, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf203 = buf196; del buf196  # reuse
        buf204 = buf195; del buf195  # reuse
        buf205 = buf194; del buf194  # reuse
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf202, buf203, buf204, buf205, 9408, 128, grid=grid(9408), stream=stream0)
        buf206 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf207 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf209 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf203, buf204, buf205, primals_221, primals_222, buf206, buf207, buf209, primals_221, primals_222, 192, 49, grid=grid(192), stream=stream0)
        del buf203
        del buf204
        del buf205
        del primals_221
        del primals_222
        buf210 = reinterpret_tensor(buf201, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf201  # reuse
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____3___act, x_101, x_103, x_88, x_92, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_24.run(buf211, buf193, buf197, buf198, primals_33, primals_34, buf202, buf206, buf207, primals_35, primals_36, buf184, buf188, buf189, primals_31, primals_32, 1204224, grid=grid(1204224), stream=stream0)
        del buf189
        del buf198
        del buf207
        del primals_32
        del primals_34
        del primals_36
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_137, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf213 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf212, buf213, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf214 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf215 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf216 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf213, buf214, buf215, buf216, 4992, 121, grid=grid(4992), stream=stream0)
        buf217 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf220 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf214, buf215, buf216, primals_224, primals_225, buf217, buf218, buf220, primals_224, primals_225, 384, 13, grid=grid(384), stream=stream0)
        del primals_224
        del primals_225
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf211, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf222 = reinterpret_tensor(buf212, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf212  # reuse
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf221, buf222, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf223 = buf216; del buf216  # reuse
        buf224 = buf215; del buf215  # reuse
        buf225 = buf214; del buf214  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf222, buf223, buf224, buf225, 4992, 121, grid=grid(4992), stream=stream0)
        buf226 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf227 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf229 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf223, buf224, buf225, primals_227, primals_228, buf226, buf227, buf229, primals_227, primals_228, 384, 13, grid=grid(384), stream=stream0)
        del primals_227
        del primals_228
        buf230 = reinterpret_tensor(buf221, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf221  # reuse
        buf231 = buf230; del buf230  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_106, x_111, x_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_28.run(buf231, buf213, buf217, buf218, primals_37, primals_38, buf222, buf226, buf227, primals_39, primals_40, 602112, grid=grid(602112), stream=stream0)
        del primals_38
        del primals_40
        buf232 = buf225; del buf225  # reuse
        buf233 = buf224; del buf224  # reuse
        buf234 = buf223; del buf223  # reuse
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf231, buf232, buf233, buf234, 4992, 121, grid=grid(4992), stream=stream0)
        buf235 = buf227; del buf227  # reuse
        buf236 = buf218; del buf218  # reuse
        buf238 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf232, buf233, buf234, primals_230, primals_231, buf235, buf236, buf238, primals_230, primals_231, 384, 13, grid=grid(384), stream=stream0)
        del primals_230
        del primals_231
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf231, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf240 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf239, buf240, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf241 = buf234; del buf234  # reuse
        buf242 = buf233; del buf233  # reuse
        buf243 = buf232; del buf232  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf240, buf241, buf242, buf243, 4992, 121, grid=grid(4992), stream=stream0)
        buf244 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf245 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf247 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf241, buf242, buf243, primals_233, primals_234, buf244, buf245, buf247, primals_233, primals_234, 384, 13, grid=grid(384), stream=stream0)
        del primals_233
        del primals_234
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf231, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf249 = reinterpret_tensor(buf239, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf239  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf248, buf249, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf250 = buf243; del buf243  # reuse
        buf251 = buf242; del buf242  # reuse
        buf252 = buf241; del buf241  # reuse
        # Source Nodes: [x_126], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf249, buf250, buf251, buf252, 4992, 121, grid=grid(4992), stream=stream0)
        buf253 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf254 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf256 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf250, buf251, buf252, primals_236, primals_237, buf253, buf254, buf256, primals_236, primals_237, 384, 13, grid=grid(384), stream=stream0)
        del primals_236
        del primals_237
        buf257 = reinterpret_tensor(buf248, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf248  # reuse
        buf258 = buf257; del buf257  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act, x_117, x_121, x_126, x_130, x_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf258, buf240, buf244, buf245, primals_43, primals_44, buf249, buf253, buf254, primals_45, primals_46, buf231, buf235, buf236, primals_41, primals_42, 602112, grid=grid(602112), stream=stream0)
        del primals_42
        del primals_44
        del primals_46
        buf259 = buf252; del buf252  # reuse
        buf260 = buf251; del buf251  # reuse
        buf261 = buf250; del buf250  # reuse
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf258, buf259, buf260, buf261, 4992, 121, grid=grid(4992), stream=stream0)
        buf262 = buf254; del buf254  # reuse
        buf263 = buf245; del buf245  # reuse
        buf265 = reinterpret_tensor(buf236, (384, ), (1, ), 0); del buf236  # reuse
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf259, buf260, buf261, primals_239, primals_240, buf262, buf263, buf265, primals_239, primals_240, 384, 13, grid=grid(384), stream=stream0)
        del primals_239
        del primals_240
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf258, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf267 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf266, buf267, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf268 = buf261; del buf261  # reuse
        buf269 = buf260; del buf260  # reuse
        buf270 = buf259; del buf259  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf267, buf268, buf269, buf270, 4992, 121, grid=grid(4992), stream=stream0)
        buf271 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf272 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf274 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf268, buf269, buf270, primals_242, primals_243, buf271, buf272, buf274, primals_242, primals_243, 384, 13, grid=grid(384), stream=stream0)
        del primals_242
        del primals_243
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf258, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf276 = reinterpret_tensor(buf266, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf266  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf275, buf276, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf277 = buf270; del buf270  # reuse
        buf278 = buf269; del buf269  # reuse
        buf279 = buf268; del buf268  # reuse
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf276, buf277, buf278, buf279, 4992, 121, grid=grid(4992), stream=stream0)
        buf280 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf281 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf283 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf277, buf278, buf279, primals_245, primals_246, buf280, buf281, buf283, primals_245, primals_246, 384, 13, grid=grid(384), stream=stream0)
        del primals_245
        del primals_246
        buf284 = reinterpret_tensor(buf275, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf275  # reuse
        buf285 = buf284; del buf284  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act, x_134, x_138, x_143, x_147, x_149], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf285, buf267, buf271, buf272, primals_49, primals_50, buf276, buf280, buf281, primals_51, primals_52, buf258, buf262, buf263, primals_47, primals_48, 602112, grid=grid(602112), stream=stream0)
        del primals_48
        del primals_50
        del primals_52
        buf286 = buf279; del buf279  # reuse
        buf287 = buf278; del buf278  # reuse
        buf288 = buf277; del buf277  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf285, buf286, buf287, buf288, 4992, 121, grid=grid(4992), stream=stream0)
        buf289 = buf281; del buf281  # reuse
        buf290 = buf272; del buf272  # reuse
        buf292 = reinterpret_tensor(buf263, (384, ), (1, ), 0); del buf263  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf286, buf287, buf288, primals_248, primals_249, buf289, buf290, buf292, primals_248, primals_249, 384, 13, grid=grid(384), stream=stream0)
        del primals_248
        del primals_249
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf285, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf294 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf293, buf294, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf295 = buf288; del buf288  # reuse
        buf296 = buf287; del buf287  # reuse
        buf297 = buf286; del buf286  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf294, buf295, buf296, buf297, 4992, 121, grid=grid(4992), stream=stream0)
        buf298 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf299 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf301 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf295, buf296, buf297, primals_251, primals_252, buf298, buf299, buf301, primals_251, primals_252, 384, 13, grid=grid(384), stream=stream0)
        del primals_251
        del primals_252
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf285, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf303 = reinterpret_tensor(buf293, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf293  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf302, buf303, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf304 = buf297; del buf297  # reuse
        buf305 = buf296; del buf296  # reuse
        buf306 = buf295; del buf295  # reuse
        # Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf303, buf304, buf305, buf306, 4992, 121, grid=grid(4992), stream=stream0)
        buf307 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf308 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf310 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf304, buf305, buf306, primals_254, primals_255, buf307, buf308, buf310, primals_254, primals_255, 384, 13, grid=grid(384), stream=stream0)
        del primals_254
        del primals_255
        buf311 = reinterpret_tensor(buf302, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf302  # reuse
        buf312 = buf311; del buf311  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act, x_151, x_155, x_160, x_164, x_166], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf312, buf294, buf298, buf299, primals_55, primals_56, buf303, buf307, buf308, primals_57, primals_58, buf285, buf289, buf290, primals_53, primals_54, 602112, grid=grid(602112), stream=stream0)
        del primals_54
        del primals_56
        del primals_58
        buf313 = buf306; del buf306  # reuse
        buf314 = buf305; del buf305  # reuse
        buf315 = buf304; del buf304  # reuse
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf312, buf313, buf314, buf315, 4992, 121, grid=grid(4992), stream=stream0)
        buf316 = buf308; del buf308  # reuse
        buf317 = buf299; del buf299  # reuse
        buf319 = reinterpret_tensor(buf290, (384, ), (1, ), 0); del buf290  # reuse
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf313, buf314, buf315, primals_257, primals_258, buf316, buf317, buf319, primals_257, primals_258, 384, 13, grid=grid(384), stream=stream0)
        del primals_257
        del primals_258
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf312, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf321 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf320, buf321, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf322 = buf315; del buf315  # reuse
        buf323 = buf314; del buf314  # reuse
        buf324 = buf313; del buf313  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf321, buf322, buf323, buf324, 4992, 121, grid=grid(4992), stream=stream0)
        buf325 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf326 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf328 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf322, buf323, buf324, primals_260, primals_261, buf325, buf326, buf328, primals_260, primals_261, 384, 13, grid=grid(384), stream=stream0)
        del primals_260
        del primals_261
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf312, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf330 = reinterpret_tensor(buf320, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf320  # reuse
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf329, buf330, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf331 = buf324; del buf324  # reuse
        buf332 = buf323; del buf323  # reuse
        buf333 = buf322; del buf322  # reuse
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf330, buf331, buf332, buf333, 4992, 121, grid=grid(4992), stream=stream0)
        buf334 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf335 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf337 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf331, buf332, buf333, primals_263, primals_264, buf334, buf335, buf337, primals_263, primals_264, 384, 13, grid=grid(384), stream=stream0)
        del primals_263
        del primals_264
        buf338 = reinterpret_tensor(buf329, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf329  # reuse
        buf339 = buf338; del buf338  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act, x_168, x_172, x_177, x_181, x_183], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf339, buf321, buf325, buf326, primals_61, primals_62, buf330, buf334, buf335, primals_63, primals_64, buf312, buf316, buf317, primals_59, primals_60, 602112, grid=grid(602112), stream=stream0)
        del primals_60
        del primals_62
        del primals_64
        buf340 = buf333; del buf333  # reuse
        buf341 = buf332; del buf332  # reuse
        buf342 = buf331; del buf331  # reuse
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf339, buf340, buf341, buf342, 4992, 121, grid=grid(4992), stream=stream0)
        buf343 = buf335; del buf335  # reuse
        buf344 = buf326; del buf326  # reuse
        buf346 = reinterpret_tensor(buf317, (384, ), (1, ), 0); del buf317  # reuse
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf340, buf341, buf342, primals_266, primals_267, buf343, buf344, buf346, primals_266, primals_267, 384, 13, grid=grid(384), stream=stream0)
        del primals_266
        del primals_267
        # Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf339, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf348 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf347, buf348, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf349 = buf342; del buf342  # reuse
        buf350 = buf341; del buf341  # reuse
        buf351 = buf340; del buf340  # reuse
        # Source Nodes: [x_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf348, buf349, buf350, buf351, 4992, 121, grid=grid(4992), stream=stream0)
        buf352 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf353 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf355 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf349, buf350, buf351, primals_269, primals_270, buf352, buf353, buf355, primals_269, primals_270, 384, 13, grid=grid(384), stream=stream0)
        del primals_269
        del primals_270
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf339, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf357 = reinterpret_tensor(buf347, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf347  # reuse
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf356, buf357, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf358 = buf351; del buf351  # reuse
        buf359 = buf350; del buf350  # reuse
        buf360 = buf349; del buf349  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf357, buf358, buf359, buf360, 4992, 121, grid=grid(4992), stream=stream0)
        buf361 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf362 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf364 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf358, buf359, buf360, primals_272, primals_273, buf361, buf362, buf364, primals_272, primals_273, 384, 13, grid=grid(384), stream=stream0)
        del primals_272
        del primals_273
        buf365 = reinterpret_tensor(buf356, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf356  # reuse
        buf366 = buf365; del buf365  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act, x_185, x_189, x_194, x_198, x_200], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf366, buf348, buf352, buf353, primals_67, primals_68, buf357, buf361, buf362, primals_69, primals_70, buf339, buf343, buf344, primals_65, primals_66, 602112, grid=grid(602112), stream=stream0)
        del primals_66
        del primals_68
        del primals_70
        buf367 = buf360; del buf360  # reuse
        buf368 = buf359; del buf359  # reuse
        buf369 = buf358; del buf358  # reuse
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf366, buf367, buf368, buf369, 4992, 121, grid=grid(4992), stream=stream0)
        buf370 = buf362; del buf362  # reuse
        buf371 = buf353; del buf353  # reuse
        buf373 = reinterpret_tensor(buf344, (384, ), (1, ), 0); del buf344  # reuse
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf367, buf368, buf369, primals_275, primals_276, buf370, buf371, buf373, primals_275, primals_276, 384, 13, grid=grid(384), stream=stream0)
        del primals_275
        del primals_276
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf366, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf375 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf374, buf375, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf376 = buf369; del buf369  # reuse
        buf377 = buf368; del buf368  # reuse
        buf378 = buf367; del buf367  # reuse
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf375, buf376, buf377, buf378, 4992, 121, grid=grid(4992), stream=stream0)
        buf379 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf380 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf382 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf376, buf377, buf378, primals_278, primals_279, buf379, buf380, buf382, primals_278, primals_279, 384, 13, grid=grid(384), stream=stream0)
        del primals_278
        del primals_279
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf366, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf384 = reinterpret_tensor(buf374, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf374  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf383, buf384, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf385 = buf378; del buf378  # reuse
        buf386 = buf377; del buf377  # reuse
        buf387 = buf376; del buf376  # reuse
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf384, buf385, buf386, buf387, 4992, 121, grid=grid(4992), stream=stream0)
        buf388 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf389 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf391 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf385, buf386, buf387, primals_281, primals_282, buf388, buf389, buf391, primals_281, primals_282, 384, 13, grid=grid(384), stream=stream0)
        del primals_281
        del primals_282
        buf392 = reinterpret_tensor(buf383, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf383  # reuse
        buf393 = buf392; del buf392  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____6___act, x_202, x_206, x_211, x_215, x_217], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf393, buf375, buf379, buf380, primals_73, primals_74, buf384, buf388, buf389, primals_75, primals_76, buf366, buf370, buf371, primals_71, primals_72, 602112, grid=grid(602112), stream=stream0)
        del primals_72
        del primals_74
        del primals_76
        buf394 = buf387; del buf387  # reuse
        buf395 = buf386; del buf386  # reuse
        buf396 = buf385; del buf385  # reuse
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf393, buf394, buf395, buf396, 4992, 121, grid=grid(4992), stream=stream0)
        buf397 = buf389; del buf389  # reuse
        buf398 = buf380; del buf380  # reuse
        buf400 = reinterpret_tensor(buf371, (384, ), (1, ), 0); del buf371  # reuse
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf394, buf395, buf396, primals_284, primals_285, buf397, buf398, buf400, primals_284, primals_285, 384, 13, grid=grid(384), stream=stream0)
        del primals_284
        del primals_285
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf393, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf402 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf401, buf402, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf403 = buf396; del buf396  # reuse
        buf404 = buf395; del buf395  # reuse
        buf405 = buf394; del buf394  # reuse
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf402, buf403, buf404, buf405, 4992, 121, grid=grid(4992), stream=stream0)
        buf406 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf407 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf409 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf403, buf404, buf405, primals_287, primals_288, buf406, buf407, buf409, primals_287, primals_288, 384, 13, grid=grid(384), stream=stream0)
        del primals_287
        del primals_288
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf393, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf411 = reinterpret_tensor(buf401, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf401  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf410, buf411, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf412 = buf405; del buf405  # reuse
        buf413 = buf404; del buf404  # reuse
        buf414 = buf403; del buf403  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf411, buf412, buf413, buf414, 4992, 121, grid=grid(4992), stream=stream0)
        buf415 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf416 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf418 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf412, buf413, buf414, primals_290, primals_291, buf415, buf416, buf418, primals_290, primals_291, 384, 13, grid=grid(384), stream=stream0)
        del primals_290
        del primals_291
        buf419 = reinterpret_tensor(buf410, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf410  # reuse
        buf420 = buf419; del buf419  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____7___act, x_219, x_223, x_228, x_232, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf420, buf402, buf406, buf407, primals_79, primals_80, buf411, buf415, buf416, primals_81, primals_82, buf393, buf397, buf398, primals_77, primals_78, 602112, grid=grid(602112), stream=stream0)
        del primals_78
        del primals_80
        del primals_82
        buf421 = buf414; del buf414  # reuse
        buf422 = buf413; del buf413  # reuse
        buf423 = buf412; del buf412  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf420, buf421, buf422, buf423, 4992, 121, grid=grid(4992), stream=stream0)
        buf424 = buf416; del buf416  # reuse
        buf425 = buf407; del buf407  # reuse
        buf427 = reinterpret_tensor(buf398, (384, ), (1, ), 0); del buf398  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf421, buf422, buf423, primals_293, primals_294, buf424, buf425, buf427, primals_293, primals_294, 384, 13, grid=grid(384), stream=stream0)
        del primals_293
        del primals_294
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf420, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf429 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf428, buf429, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf430 = buf423; del buf423  # reuse
        buf431 = buf422; del buf422  # reuse
        buf432 = buf421; del buf421  # reuse
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf429, buf430, buf431, buf432, 4992, 121, grid=grid(4992), stream=stream0)
        buf433 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf436 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf430, buf431, buf432, primals_296, primals_297, buf433, buf434, buf436, primals_296, primals_297, 384, 13, grid=grid(384), stream=stream0)
        del primals_296
        del primals_297
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf420, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf438 = reinterpret_tensor(buf428, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf428  # reuse
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf437, buf438, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf439 = buf432; del buf432  # reuse
        buf440 = buf431; del buf431  # reuse
        buf441 = buf430; del buf430  # reuse
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf438, buf439, buf440, buf441, 4992, 121, grid=grid(4992), stream=stream0)
        buf442 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf443 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf445 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf439, buf440, buf441, primals_299, primals_300, buf442, buf443, buf445, primals_299, primals_300, 384, 13, grid=grid(384), stream=stream0)
        del primals_299
        del primals_300
        buf446 = reinterpret_tensor(buf437, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf437  # reuse
        buf447 = buf446; del buf446  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____8___act, x_236, x_240, x_245, x_249, x_251], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf447, buf429, buf433, buf434, primals_85, primals_86, buf438, buf442, buf443, primals_87, primals_88, buf420, buf424, buf425, primals_83, primals_84, 602112, grid=grid(602112), stream=stream0)
        del primals_84
        del primals_86
        del primals_88
        buf448 = buf441; del buf441  # reuse
        buf449 = buf440; del buf440  # reuse
        buf450 = buf439; del buf439  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf447, buf448, buf449, buf450, 4992, 121, grid=grid(4992), stream=stream0)
        buf451 = buf443; del buf443  # reuse
        buf452 = buf434; del buf434  # reuse
        buf454 = reinterpret_tensor(buf425, (384, ), (1, ), 0); del buf425  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf448, buf449, buf450, primals_302, primals_303, buf451, buf452, buf454, primals_302, primals_303, 384, 13, grid=grid(384), stream=stream0)
        del primals_302
        del primals_303
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf447, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf456 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf455, buf456, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf457 = buf450; del buf450  # reuse
        buf458 = buf449; del buf449  # reuse
        buf459 = buf448; del buf448  # reuse
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf456, buf457, buf458, buf459, 4992, 121, grid=grid(4992), stream=stream0)
        buf460 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf461 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf463 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf457, buf458, buf459, primals_305, primals_306, buf460, buf461, buf463, primals_305, primals_306, 384, 13, grid=grid(384), stream=stream0)
        del primals_305
        del primals_306
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf447, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf465 = reinterpret_tensor(buf455, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf455  # reuse
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf464, buf465, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf466 = buf459; del buf459  # reuse
        buf467 = buf458; del buf458  # reuse
        buf468 = buf457; del buf457  # reuse
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf465, buf466, buf467, buf468, 4992, 121, grid=grid(4992), stream=stream0)
        buf469 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf470 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf472 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf466, buf467, buf468, primals_308, primals_309, buf469, buf470, buf472, primals_308, primals_309, 384, 13, grid=grid(384), stream=stream0)
        del primals_308
        del primals_309
        buf473 = reinterpret_tensor(buf464, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf464  # reuse
        buf474 = buf473; del buf473  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____9___act, x_253, x_257, x_262, x_266, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf474, buf456, buf460, buf461, primals_91, primals_92, buf465, buf469, buf470, primals_93, primals_94, buf447, buf451, buf452, primals_89, primals_90, 602112, grid=grid(602112), stream=stream0)
        del primals_90
        del primals_92
        del primals_94
        buf475 = buf468; del buf468  # reuse
        buf476 = buf467; del buf467  # reuse
        buf477 = buf466; del buf466  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf474, buf475, buf476, buf477, 4992, 121, grid=grid(4992), stream=stream0)
        buf478 = buf470; del buf470  # reuse
        buf479 = buf461; del buf461  # reuse
        buf481 = reinterpret_tensor(buf452, (384, ), (1, ), 0); del buf452  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf475, buf476, buf477, primals_311, primals_312, buf478, buf479, buf481, primals_311, primals_312, 384, 13, grid=grid(384), stream=stream0)
        del primals_311
        del primals_312
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf474, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf483 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf482, buf483, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf484 = buf477; del buf477  # reuse
        buf485 = buf476; del buf476  # reuse
        buf486 = buf475; del buf475  # reuse
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf483, buf484, buf485, buf486, 4992, 121, grid=grid(4992), stream=stream0)
        buf487 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf488 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf490 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf484, buf485, buf486, primals_314, primals_315, buf487, buf488, buf490, primals_314, primals_315, 384, 13, grid=grid(384), stream=stream0)
        del primals_314
        del primals_315
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf474, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf492 = reinterpret_tensor(buf482, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf482  # reuse
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf491, buf492, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf493 = buf486; del buf486  # reuse
        buf494 = buf485; del buf485  # reuse
        buf495 = buf484; del buf484  # reuse
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf492, buf493, buf494, buf495, 4992, 121, grid=grid(4992), stream=stream0)
        buf496 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf497 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf499 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf493, buf494, buf495, primals_317, primals_318, buf496, buf497, buf499, primals_317, primals_318, 384, 13, grid=grid(384), stream=stream0)
        del primals_317
        del primals_318
        buf500 = reinterpret_tensor(buf491, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf491  # reuse
        buf501 = buf500; del buf500  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____10___act, x_270, x_274, x_279, x_283, x_285], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf501, buf483, buf487, buf488, primals_97, primals_98, buf492, buf496, buf497, primals_99, primals_100, buf474, buf478, buf479, primals_95, primals_96, 602112, grid=grid(602112), stream=stream0)
        del primals_100
        del primals_96
        del primals_98
        buf502 = buf495; del buf495  # reuse
        buf503 = buf494; del buf494  # reuse
        buf504 = buf493; del buf493  # reuse
        # Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf501, buf502, buf503, buf504, 4992, 121, grid=grid(4992), stream=stream0)
        buf505 = buf497; del buf497  # reuse
        buf506 = buf488; del buf488  # reuse
        buf508 = reinterpret_tensor(buf479, (384, ), (1, ), 0); del buf479  # reuse
        # Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf502, buf503, buf504, primals_320, primals_321, buf505, buf506, buf508, primals_320, primals_321, 384, 13, grid=grid(384), stream=stream0)
        del primals_320
        del primals_321
        # Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf501, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf510 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf509, buf510, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf511 = buf504; del buf504  # reuse
        buf512 = buf503; del buf503  # reuse
        buf513 = buf502; del buf502  # reuse
        # Source Nodes: [x_291], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf510, buf511, buf512, buf513, 4992, 121, grid=grid(4992), stream=stream0)
        buf514 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf515 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf517 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf511, buf512, buf513, primals_323, primals_324, buf514, buf515, buf517, primals_323, primals_324, 384, 13, grid=grid(384), stream=stream0)
        del primals_323
        del primals_324
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf501, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf519 = reinterpret_tensor(buf509, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf509  # reuse
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf518, buf519, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf520 = buf513; del buf513  # reuse
        buf521 = buf512; del buf512  # reuse
        buf522 = buf511; del buf511  # reuse
        # Source Nodes: [x_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf519, buf520, buf521, buf522, 4992, 121, grid=grid(4992), stream=stream0)
        buf523 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf524 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf526 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf520, buf521, buf522, primals_326, primals_327, buf523, buf524, buf526, primals_326, primals_327, 384, 13, grid=grid(384), stream=stream0)
        del primals_326
        del primals_327
        buf527 = reinterpret_tensor(buf518, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf518  # reuse
        buf528 = buf527; del buf527  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____11___act, x_287, x_291, x_296, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf528, buf510, buf514, buf515, primals_103, primals_104, buf519, buf523, buf524, primals_105, primals_106, buf501, buf505, buf506, primals_101, primals_102, 602112, grid=grid(602112), stream=stream0)
        del primals_102
        del primals_104
        del primals_106
        buf529 = buf522; del buf522  # reuse
        buf530 = buf521; del buf521  # reuse
        buf531 = buf520; del buf520  # reuse
        # Source Nodes: [x_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf528, buf529, buf530, buf531, 4992, 121, grid=grid(4992), stream=stream0)
        buf532 = buf524; del buf524  # reuse
        buf533 = buf515; del buf515  # reuse
        buf535 = reinterpret_tensor(buf506, (384, ), (1, ), 0); del buf506  # reuse
        # Source Nodes: [x_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf529, buf530, buf531, primals_329, primals_330, buf532, buf533, buf535, primals_329, primals_330, 384, 13, grid=grid(384), stream=stream0)
        del primals_329
        del primals_330
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf528, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf537 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf536, buf537, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf538 = buf531; del buf531  # reuse
        buf539 = buf530; del buf530  # reuse
        buf540 = buf529; del buf529  # reuse
        # Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf537, buf538, buf539, buf540, 4992, 121, grid=grid(4992), stream=stream0)
        buf541 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf542 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf544 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf538, buf539, buf540, primals_332, primals_333, buf541, buf542, buf544, primals_332, primals_333, 384, 13, grid=grid(384), stream=stream0)
        del primals_332
        del primals_333
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf528, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf546 = reinterpret_tensor(buf536, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf536  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf545, buf546, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf547 = buf540; del buf540  # reuse
        buf548 = buf539; del buf539  # reuse
        buf549 = buf538; del buf538  # reuse
        # Source Nodes: [x_313], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf546, buf547, buf548, buf549, 4992, 121, grid=grid(4992), stream=stream0)
        buf550 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf551 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf553 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_313], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf547, buf548, buf549, primals_335, primals_336, buf550, buf551, buf553, primals_335, primals_336, 384, 13, grid=grid(384), stream=stream0)
        del primals_335
        del primals_336
        buf554 = reinterpret_tensor(buf545, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf545  # reuse
        buf555 = buf554; del buf554  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____12___act, x_304, x_308, x_313, x_317, x_319], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf555, buf537, buf541, buf542, primals_109, primals_110, buf546, buf550, buf551, primals_111, primals_112, buf528, buf532, buf533, primals_107, primals_108, 602112, grid=grid(602112), stream=stream0)
        del primals_108
        del primals_110
        del primals_112
        buf556 = buf549; del buf549  # reuse
        buf557 = buf548; del buf548  # reuse
        buf558 = buf547; del buf547  # reuse
        # Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf555, buf556, buf557, buf558, 4992, 121, grid=grid(4992), stream=stream0)
        buf559 = buf551; del buf551  # reuse
        buf560 = buf542; del buf542  # reuse
        buf562 = reinterpret_tensor(buf533, (384, ), (1, ), 0); del buf533  # reuse
        # Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf556, buf557, buf558, primals_338, primals_339, buf559, buf560, buf562, primals_338, primals_339, 384, 13, grid=grid(384), stream=stream0)
        del primals_338
        del primals_339
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf555, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf564 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf563, buf564, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf565 = buf558; del buf558  # reuse
        buf566 = buf557; del buf557  # reuse
        buf567 = buf556; del buf556  # reuse
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf564, buf565, buf566, buf567, 4992, 121, grid=grid(4992), stream=stream0)
        buf568 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf569 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf571 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf565, buf566, buf567, primals_341, primals_342, buf568, buf569, buf571, primals_341, primals_342, 384, 13, grid=grid(384), stream=stream0)
        del primals_341
        del primals_342
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf555, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf573 = reinterpret_tensor(buf563, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf563  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf572, buf573, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf574 = buf567; del buf567  # reuse
        buf575 = buf566; del buf566  # reuse
        buf576 = buf565; del buf565  # reuse
        # Source Nodes: [x_330], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf573, buf574, buf575, buf576, 4992, 121, grid=grid(4992), stream=stream0)
        buf577 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf578 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf580 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_330], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf574, buf575, buf576, primals_344, primals_345, buf577, buf578, buf580, primals_344, primals_345, 384, 13, grid=grid(384), stream=stream0)
        del buf574
        del buf575
        del buf576
        del primals_344
        del primals_345
        buf581 = reinterpret_tensor(buf572, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf572  # reuse
        buf582 = buf581; del buf581  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____13___act, x_321, x_325, x_330, x_334, x_336], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf582, buf564, buf568, buf569, primals_115, primals_116, buf573, buf577, buf578, primals_117, primals_118, buf555, buf559, buf560, primals_113, primals_114, 602112, grid=grid(602112), stream=stream0)
        del buf560
        del buf569
        del buf578
        del primals_114
        del primals_116
        del primals_118
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, primals_165, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (8, 1408, 7, 7), (68992, 49, 7, 1))
        buf584 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf583, buf584, 11264, 49, grid=grid(11264, 49), stream=stream0)
        buf585 = empty_strided((1, 1408, 1, 1, 4), (5632, 1, 5632, 5632, 1408), device='cuda', dtype=torch.float32)
        buf586 = empty_strided((1, 1408, 1, 1, 4), (5632, 1, 5632, 5632, 1408), device='cuda', dtype=torch.float32)
        buf587 = empty_strided((1, 1408, 1, 1, 4), (5632, 1, 5632, 5632, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf584, buf585, buf586, buf587, 5632, 98, grid=grid(5632), stream=stream0)
        buf588 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf589 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf591 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_32.run(buf585, buf586, buf587, primals_347, primals_348, buf588, buf589, buf591, primals_347, primals_348, 1408, 4, grid=grid(1408), stream=stream0)
        del primals_347
        del primals_348
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf582, buf21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 1408, 7, 7), (68992, 49, 7, 1))
        buf593 = reinterpret_tensor(buf583, (8, 1408, 7, 7), (68992, 1, 9856, 1408), 0); del buf583  # reuse
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf592, buf593, 11264, 49, grid=grid(11264, 49), stream=stream0)
        buf594 = buf587; del buf587  # reuse
        buf595 = buf586; del buf586  # reuse
        buf596 = buf585; del buf585  # reuse
        # Source Nodes: [x_344], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf593, buf594, buf595, buf596, 5632, 98, grid=grid(5632), stream=stream0)
        buf597 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf598 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf600 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_344], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_32.run(buf594, buf595, buf596, primals_350, primals_351, buf597, buf598, buf600, primals_350, primals_351, 1408, 4, grid=grid(1408), stream=stream0)
        del buf594
        del buf595
        del buf596
        del primals_350
        del primals_351
        buf601 = reinterpret_tensor(buf592, (8, 1408, 7, 7), (68992, 1, 9856, 1408), 0); del buf592  # reuse
        buf605 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_339, x_344, x_348, x_350], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_33.run(buf584, buf588, buf589, primals_119, primals_120, buf593, buf597, buf598, primals_121, primals_122, buf601, buf605, 551936, grid=grid(551936), stream=stream0)
        del buf589
        del buf598
        del primals_120
        del primals_122
        buf602 = empty_strided((8, 1408, 1, 1), (1408, 1, 11264, 11264), device='cuda', dtype=torch.float32)
        buf603 = reinterpret_tensor(buf602, (8, 1408), (1408, 1), 0); del buf602  # reuse
        # Source Nodes: [x_350, x_353, x_355], Original ATen: [aten.mean, aten.relu, aten.view]
        triton_per_fused_mean_relu_view_34.run(buf603, buf601, 11264, 49, grid=grid(11264), stream=stream0)
        del buf601
        buf604 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_168, buf603, reinterpret_tensor(primals_167, (1408, 1000), (1, 1408), 0), alpha=1, beta=1, out=buf604)
        del primals_168
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_169, primals_169, 1, grid=grid(1), stream=stream0)
        del primals_169
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_172, primals_172, 1, grid=grid(1), stream=stream0)
        del primals_172
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_175, primals_175, 1, grid=grid(1), stream=stream0)
        del primals_175
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_178, primals_178, 1, grid=grid(1), stream=stream0)
        del primals_178
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_181, primals_181, 1, grid=grid(1), stream=stream0)
        del primals_181
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_184, primals_184, 1, grid=grid(1), stream=stream0)
        del primals_184
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_187, primals_187, 1, grid=grid(1), stream=stream0)
        del primals_187
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_190, primals_190, 1, grid=grid(1), stream=stream0)
        del primals_190
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_193, primals_193, 1, grid=grid(1), stream=stream0)
        del primals_193
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_196, primals_196, 1, grid=grid(1), stream=stream0)
        del primals_196
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_199, primals_199, 1, grid=grid(1), stream=stream0)
        del primals_199
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_202, primals_202, 1, grid=grid(1), stream=stream0)
        del primals_202
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_205, primals_205, 1, grid=grid(1), stream=stream0)
        del primals_205
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_208, primals_208, 1, grid=grid(1), stream=stream0)
        del primals_208
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_211, primals_211, 1, grid=grid(1), stream=stream0)
        del primals_211
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_214, primals_214, 1, grid=grid(1), stream=stream0)
        del primals_214
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_217, primals_217, 1, grid=grid(1), stream=stream0)
        del primals_217
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_220, primals_220, 1, grid=grid(1), stream=stream0)
        del primals_220
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_223, primals_223, 1, grid=grid(1), stream=stream0)
        del primals_223
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_226, primals_226, 1, grid=grid(1), stream=stream0)
        del primals_226
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_229, primals_229, 1, grid=grid(1), stream=stream0)
        del primals_229
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_232, primals_232, 1, grid=grid(1), stream=stream0)
        del primals_232
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_235, primals_235, 1, grid=grid(1), stream=stream0)
        del primals_235
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_238, primals_238, 1, grid=grid(1), stream=stream0)
        del primals_238
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_241, primals_241, 1, grid=grid(1), stream=stream0)
        del primals_241
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_244, primals_244, 1, grid=grid(1), stream=stream0)
        del primals_244
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_247, primals_247, 1, grid=grid(1), stream=stream0)
        del primals_247
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_250, primals_250, 1, grid=grid(1), stream=stream0)
        del primals_250
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_253, primals_253, 1, grid=grid(1), stream=stream0)
        del primals_253
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_256, primals_256, 1, grid=grid(1), stream=stream0)
        del primals_256
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_259, primals_259, 1, grid=grid(1), stream=stream0)
        del primals_259
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_262, primals_262, 1, grid=grid(1), stream=stream0)
        del primals_262
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_319, primals_319, 1, grid=grid(1), stream=stream0)
        del primals_319
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_322, primals_322, 1, grid=grid(1), stream=stream0)
        del primals_322
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_325, primals_325, 1, grid=grid(1), stream=stream0)
        del primals_325
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_328, primals_328, 1, grid=grid(1), stream=stream0)
        del primals_328
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_331, primals_331, 1, grid=grid(1), stream=stream0)
        del primals_331
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_334, primals_334, 1, grid=grid(1), stream=stream0)
        del primals_334
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        return (buf604, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, buf0, primals_125, buf1, primals_127, buf2, primals_129, buf3, primals_131, buf4, primals_133, buf5, primals_135, buf6, primals_137, buf7, primals_139, buf8, primals_141, buf9, primals_143, buf10, primals_145, buf11, primals_147, buf12, primals_149, buf13, primals_151, buf14, primals_153, buf15, primals_155, buf16, primals_157, buf17, primals_159, buf18, primals_161, buf19, primals_163, buf20, primals_165, buf21, buf22, buf24, buf34, buf36, buf46, buf48, buf50, buf60, buf62, buf72, buf74, buf84, buf86, buf96, buf98, buf108, buf110, buf112, buf119, buf121, buf128, buf130, buf137, buf139, buf146, buf148, buf155, buf157, buf164, buf166, buf173, buf175, buf182, buf184, buf191, buf193, buf200, buf202, buf209, buf211, buf213, buf220, buf222, buf229, buf231, buf238, buf240, buf247, buf249, buf256, buf258, buf265, buf267, buf274, buf276, buf283, buf285, buf292, buf294, buf301, buf303, buf310, buf312, buf319, buf321, buf328, buf330, buf337, buf339, buf346, buf348, buf355, buf357, buf364, buf366, buf373, buf375, buf382, buf384, buf391, buf393, buf400, buf402, buf409, buf411, buf418, buf420, buf427, buf429, buf436, buf438, buf445, buf447, buf454, buf456, buf463, buf465, buf472, buf474, buf481, buf483, buf490, buf492, buf499, buf501, buf508, buf510, buf517, buf519, buf526, buf528, buf535, buf537, buf544, buf546, buf553, buf555, buf562, buf564, buf571, buf573, buf580, buf582, buf584, buf591, buf593, buf600, buf603, reinterpret_tensor(primals_167, (1000, 1408), (1408, 1), 0), buf605, reinterpret_tensor(buf597, (1, 1408, 1, 1), (1408, 1, 1, 1), 0), reinterpret_tensor(buf588, (1, 1408, 1, 1), (1408, 1, 1, 1), 0), reinterpret_tensor(buf577, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf568, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf559, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf550, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf541, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf532, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf523, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf514, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf505, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf496, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf487, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf478, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf469, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf460, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf451, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf442, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf433, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf424, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf415, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf388, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf379, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf370, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf361, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf352, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf343, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf334, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf325, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf316, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf307, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf298, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf289, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf280, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf271, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf262, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf253, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf244, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf235, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf226, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf217, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf206, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf197, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf188, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf179, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf170, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf161, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf152, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf143, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf134, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf125, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf116, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf105, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf69, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf43, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf31, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1408, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_170 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_176 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_179 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_182 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_185 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_188 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_191 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_194 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_209 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_215 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_233 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_236 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_248 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_251 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_254 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_260 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_263 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_320 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_323 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_329 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_335 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
