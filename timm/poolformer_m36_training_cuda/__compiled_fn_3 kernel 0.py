
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


# kernel path: /tmp/torchinductor_youkaichao/qu/cqufoffypa4zdtvus2kwpvg24ynfd4qurzieeqiprzqyhbces2d3.py
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
    size_hints=[512, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/hy/chyksb4sdyzhiz42kw6vyd6atrr32f77s2e4luxgmqgwxbf2zfam.py
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
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtdtqkz3a2cf5qnbewd6xewr7f52pqqkmslcbxiiyzjjr5j4ach.py
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
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
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


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zy77nj6tcqsz3llgsoiffa3sfxbhxhev736n5gyv2gppyglta3.py
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
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/e6/ce63awra6riwu4j3u7kiwqmeillggb66zlncojz3bremkikvudsw.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjmbu5jt6cpplbo3mmvait7biuz46gnujfqdz54wc5rfjw36eog.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_per_fused_native_group_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 37
    x2 = (xindex // 2368)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 8137, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (8137*x1)
    tmp4 = tl.full([1, 1], 301056, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp23 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp22, 0)
    tmp27 = tl.where(rmask & xmask, tmp23, 0)
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26, tmp27, tmp28, 1)
    tmp32 = tmp29[:, None]
    tmp33 = tmp30[:, None]
    tmp34 = tmp31[:, None]
    tl.store(out_ptr0 + (x5), tmp32, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
    tl.store(out_ptr2 + (x5), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxgqffefwksdf5hiuq5f3bhfxggg2j7xdk6okg4ef3l2ohmaoa7.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_per_fused_native_group_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprbad5fqryl57pq2fg5rswg7tpplpnerhrojrunma736xugapxf.py
# Source Nodes: [group_norm], Original ATen: [aten.detach, aten.native_group_norm]
# group_norm => var_mean
triton_per_fused_detach_native_group_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_detach_native_group_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 37
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (37*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 301056.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caewnbmenv5vxre7voj7ttmeg4ws5gyzrzjlp6yihbbizy345hmu.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => add_1, mul_1
triton_poi_fused_native_group_norm_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 301056)
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 301056.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvc4xaqlv64fxf34rl7pzhjctuyyithb4amd3jm43vdgwtvirw46.py
# Source Nodes: [y], Original ATen: [aten.avg_pool2d]
# y => avg_pool2d
triton_poi_fused_avg_pool2d_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 56
    x1 = (xindex // 96) % 56
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5472) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-5376) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-5280) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-96) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (96 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (5280 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (5376 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (5472 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tl.store(out_ptr0 + (x6), tmp89, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/crehkj2aultzcy6w6hsd5roq4rompuw7unbjkf46ohyggkfvdqty.py
# Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
# group_norm_1 => var_mean_1
triton_per_fused_native_group_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 37
    x2 = (xindex // 2368)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 8137, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (8137*x1)
    tmp4 = tl.full([1, 1], 301056, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugyqxpixp5cqj6ncbz4n4xlicerrpxzcq64qa7mf3hoxdmsjvvv.py
# Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
# group_norm_1 => add_4, mul_4
triton_poi_fused_native_group_norm_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 301056.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgtcnjyhy65pivf2h2l6a5xw5tg4ubvv7lry4tawwmzbimzrpqk.py
# Source Nodes: [x_5], Original ATen: [aten.convolution]
# x_5 => convolution_1
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (384*x2) + (1204224*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cup3kjmgf4dus4ufcgt6rrjytlnjkvpjqsjy37ee6agyxasggrsx.py
# Source Nodes: [x_6], Original ATen: [aten.gelu]
# x_6 => add_5, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f7/cf7dkt5pws7g5pmqabls6fnqpmppev6ydp4tunzjphxwkkgo6ymz.py
# Source Nodes: [mul, mul_1, sub, x_11, x_4], Original ATen: [aten.add, aten.mul, aten.sub]
# mul => mul_2
# mul_1 => mul_8
# sub => sub_1
# x_11 => add_6
# x_4 => add_2
triton_poi_fused_add_mul_sub_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (y0 + (3136*x2) + (301056*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctf4she6tbi356qnk4xpxr3gu74uocz6r6vxxqahkciob4stdh5p.py
# Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# group_norm_2 => var_mean_2
triton_red_fused_native_group_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 8137
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 37
    x1 = (xindex // 37)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8137*x0)
        tmp1 = tl.full([1, 1], 301056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pvfi47dexinb6qqw6es2reimkncyfbjup54ve6ndh26twpag6b.py
# Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# group_norm_2 => add_8, mul_10
triton_poi_fused_native_group_norm_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 96)
    y0 = yindex % 96
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 301056.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklzerzu7p6ovyqson6hy5ismmak6qczlktq5u4uou44ujamldgu.py
# Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
# group_norm_3 => var_mean_3
triton_per_fused_native_group_norm_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 37
    x2 = (xindex // 2368)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 8137, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (8137*x1)
    tmp4 = tl.full([1, 1], 301056, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((301056*x2) + ((r3 + (128*x0) + (8137*x1)) % 301056)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvva4unxvcyculcdh36omzsym32udvagiqswvd6igr7r5hgfvfn.py
# Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
# group_norm_3 => add_11, mul_13
triton_poi_fused_native_group_norm_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 301056.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg64nrnj6sa7xakzhigibcqzgk4mg3romjwewzpaktnonmk72kli.py
# Source Nodes: [mul_2, mul_3, sub_1, x_12, x_17, x_19], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
# mul_2 => mul_11
# mul_3 => mul_17
# sub_1 => sub_4
# x_12 => add_9
# x_17 => convolution_4
# x_19 => add_13
triton_poi_fused_add_convolution_mul_sub_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sub_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp2, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhggpggx3jufumx322xdgcgsbxvjsz2fe3usfwqduq65l4ip32a.py
# Source Nodes: [mul_10, mul_11, sub_5, x_44, x_52], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_10 => mul_47
# mul_11 => mul_53
# sub_5 => sub_16
# x_44 => add_37
# x_52 => add_41
triton_poi_fused_add_mul_sub_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cys5mpa7rt547m7qwc22eci7aqhtj3p3rmwgopdqvz2pciu2nbcs.py
# Source Nodes: [x_55], Original ATen: [aten.convolution]
# x_55 => convolution_13
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ykh2rkmqyjfxxilxrvegps3apxqq5g6esb32o5x2lsefugypne.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => var_mean_12
triton_per_fused_native_group_norm_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 19) % 62
    x0 = xindex % 19
    x2 = (xindex // 1178)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7923*x0)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp23 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp22, 0)
    tmp27 = tl.where(rmask & xmask, tmp23, 0)
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26, tmp27, tmp28, 1)
    tmp32 = tmp29[:, None]
    tmp33 = tmp30[:, None]
    tmp34 = tmp31[:, None]
    tl.store(out_ptr0 + (x4), tmp32, xmask)
    tl.store(out_ptr1 + (x4), tmp33, xmask)
    tl.store(out_ptr2 + (x4), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzw5fyjo2ccynrch7zkdoqpb5jeestbcw4wdgouvhwptngwez2d.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => var_mean_12
triton_per_fused_native_group_norm_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 62
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 19
    x1 = (xindex // 19)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (19*r2) + (1178*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (19*r2) + (1178*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (19*r2) + (1178*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/coj7m2ow2y5qp5knsiu27pdyhkwefba6gvqganwxjlgvim5tqtxy.py
# Source Nodes: [group_norm_12], Original ATen: [aten.detach, aten.native_group_norm]
# group_norm_12 => var_mean_12
triton_per_fused_detach_native_group_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_detach_native_group_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (19*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 150528.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggglgt37btdxdhnoxyky5neuenuq4gcag7aey6ruj4f5oyeodfr.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => add_43, mul_55
triton_poi_fused_native_group_norm_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 150528)
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 150528.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kpqt77ldxpcthvjzxqvepjmplub6n5fbzbapjjo5hhm3wpozal.py
# Source Nodes: [y_6], Original ATen: [aten.avg_pool2d]
# y_6 => avg_pool2d_6
triton_poi_fused_avg_pool2d_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 28
    x1 = (xindex // 192) % 28
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5568) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-5376) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-5184) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-192) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (192 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (5184 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (5376 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (5568 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tl.store(out_ptr0 + (x6), tmp89, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbwjlxmc2pglecm4fhkbcrjidykhnnctyxgai2jcsinktlmf5xe.py
# Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# group_norm_13 => var_mean_13
triton_per_fused_native_group_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 19) % 62
    x0 = xindex % 19
    x2 = (xindex // 1178)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7923*x0)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x1) + (7923*x0)) // 784) % 192), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x4), tmp38, xmask)
    tl.store(out_ptr1 + (x4), tmp39, xmask)
    tl.store(out_ptr2 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs33jf5zzdl26hyrlw2xryrd2zeyr5g5y5nqytevztsa5wej7dbv.py
# Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# group_norm_13 => add_46, mul_58
triton_poi_fused_native_group_norm_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 192
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 150528.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa2q2g5vmgcrhz2bcivxizgg27qas6njgxvqsjesuvnibysfn7l.py
# Source Nodes: [x_57], Original ATen: [aten.convolution]
# x_57 => convolution_14
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (768*x2) + (602112*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgjibwwbzm7enje5plpw4vh6ymcwm55hw64fmdwzte2hrf5x57q.py
# Source Nodes: [x_58], Original ATen: [aten.gelu]
# x_58 => add_47, erf_6, mul_59, mul_60, mul_61
triton_poi_fused_gelu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbkrxj2i7noqh7nkxs7vpsmuskvhkk3dx6ion7dhepj536jzbw3.py
# Source Nodes: [mul_12, mul_13, sub_6, x_56, x_63], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_12 => mul_56
# mul_13 => mul_62
# sub_6 => sub_19
# x_56 => add_44
# x_63 => add_48
triton_poi_fused_add_mul_sub_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckwlyuzx6yigljunltndlhola3qg72seajqmkozca23rza2mq4e.py
# Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
# group_norm_14 => var_mean_14
triton_red_fused_native_group_norm_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 7923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7923*x0)
        tmp1 = tl.full([1, 1], 150528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5l7k5i2tnjcd5khujlmgcnivsf6633xw4y6nwlbe7mshin2exra.py
# Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
# group_norm_14 => add_50, mul_64
triton_poi_fused_native_group_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 192)
    y0 = yindex % 192
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 150528.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxvz7y4l3teoau3ie5b7wepqzb7lv4ho6w4k5z3fno36u23cpjj.py
# Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
# group_norm_15 => var_mean_15
triton_per_fused_native_group_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 19) % 62
    x0 = xindex % 19
    x2 = (xindex // 1178)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7923*x0)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((150528*x2) + ((r3 + (128*x1) + (7923*x0)) % 150528)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x1) + (7923*x0)) // 784) % 192), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x4), tmp38, xmask)
    tl.store(out_ptr1 + (x4), tmp39, xmask)
    tl.store(out_ptr2 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5ih6xlvwhupni56wy3cxfangwafzlmzf3zqjtzhw5it7bvhtgfp.py
# Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
# group_norm_15 => add_53, mul_67
triton_poi_fused_native_group_norm_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 150528.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cs/ccs2ignybvfjv2jjwbl3e5xafzxe65msjrqkbum2dfo6yumxzvw7.py
# Source Nodes: [mul_14, mul_15, sub_7, x_64, x_69, x_71], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
# mul_14 => mul_65
# mul_15 => mul_71
# sub_7 => sub_22
# x_64 => add_51
# x_69 => convolution_17
# x_71 => add_55
triton_poi_fused_add_convolution_mul_sub_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sub_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp2, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coym7duxbl5tkgexljaqf2pdrvew2mzelv2vazt7opmfxjkweux4.py
# Source Nodes: [mul_22, mul_23, sub_11, x_104, x_96], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_22 => mul_101
# mul_23 => mul_107
# sub_11 => sub_34
# x_104 => add_83
# x_96 => add_79
triton_poi_fused_add_mul_sub_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbzcou7a5iv7hcplxuzr3gvpynox2lhhsvujgnhyrqca7nasmrw.py
# Source Nodes: [x_107], Original ATen: [aten.convolution]
# x_107 => convolution_26
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhcc3ts7plsszgf4do7lrcrprumua7few3gwi3x3aofjewcjrq3.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => var_mean_24
triton_per_fused_native_group_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp23 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp22, 0)
    tmp27 = tl.where(rmask & xmask, tmp23, 0)
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26, tmp27, tmp28, 1)
    tmp32 = tmp29[:, None]
    tmp33 = tmp30[:, None]
    tmp34 = tmp31[:, None]
    tl.store(out_ptr0 + (x4), tmp32, xmask)
    tl.store(out_ptr1 + (x4), tmp33, xmask)
    tl.store(out_ptr2 + (x4), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2ciyuk7d7eev322kl7ne5gzsbgqeqaefnaw2ogizihpapwfges4.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => var_mean_24
triton_per_fused_native_group_norm_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 59
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 10
    x1 = (xindex // 10)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfwbnvhnfdzdlxbnuhvglespyrum55ijdyvnu7hp6jx4fvzoczp.py
# Source Nodes: [group_norm_24], Original ATen: [aten.detach, aten.native_group_norm]
# group_norm_24 => var_mean_24
triton_per_fused_detach_native_group_norm_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_detach_native_group_norm_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (10*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 75264.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qv4vns3wdxh5snrdcdafyyhm2x5ip7hoy5zqotudkixxjqamub.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => add_85, mul_109
triton_poi_fused_native_group_norm_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 75264)
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 75264.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3sbaspq7kxhijymttz2rdeu76zfjeenjppnuf3k5wykvfrisyws.py
# Source Nodes: [y_12], Original ATen: [aten.avg_pool2d]
# y_12 => avg_pool2d_12
triton_poi_fused_avg_pool2d_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 14
    x1 = (xindex // 384) % 14
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5760) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-5376) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-4992) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-384) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (384 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (4992 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (5376 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (5760 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tl.store(out_ptr0 + (x6), tmp89, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqq4jfyhhd2lvvu2omraax2ibamluw6z3b2i2r4hjdxcivfxeu5m.py
# Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
# group_norm_25 => var_mean_25
triton_per_fused_native_group_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x1) + (7527*x0)) // 196) % 384), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x4), tmp38, xmask)
    tl.store(out_ptr1 + (x4), tmp39, xmask)
    tl.store(out_ptr2 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aa/caayykbmgslcha3uoecrfhf72tyuftpahlshk6ct27vhgdxl5phg.py
# Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
# group_norm_25 => add_88, mul_112
triton_poi_fused_native_group_norm_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 75264.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx672spkopyjaftfjlpkke5pduic35nmpxg7dcxp3srfxqpgazie.py
# Source Nodes: [x_109], Original ATen: [aten.convolution]
# x_109 => convolution_27
triton_poi_fused_convolution_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (1536*x2) + (301056*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmi2uj3z3oqbmrqdypmnrzmuxz3pe6jnxva2ha37owhxi7asgnsu.py
# Source Nodes: [x_110], Original ATen: [aten.gelu]
# x_110 => add_89, erf_12, mul_113, mul_114, mul_115
triton_poi_fused_gelu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxdfc6y74fjjmkaei4lapzuwpmztvlpgrqz7z66tnfsz7r6nyus.py
# Source Nodes: [mul_24, mul_25, sub_12, x_108, x_115], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_24 => mul_110
# mul_25 => mul_116
# sub_12 => sub_37
# x_108 => add_86
# x_115 => add_90
triton_poi_fused_add_mul_sub_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkjvm6udunqcld2ukmxm6bg5m5pfmofo5vix7ap3d6vmx5hsasi.py
# Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
# group_norm_26 => var_mean_26
triton_red_fused_native_group_norm_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 75264, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j4/cj4bn7k2zxyibnrpjya57ttvjjaipismnwsriyjtv6abpnl6k3cx.py
# Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
# group_norm_26 => add_92, mul_118
triton_poi_fused_native_group_norm_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 384)
    y0 = yindex % 384
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 75264.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvpjij63zeymzhglezzijy6srheipy5bxoqkz4c3qle7xd6635p.py
# Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
# group_norm_27 => var_mean_27
triton_per_fused_native_group_norm_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((75264*x2) + ((r3 + (128*x1) + (7527*x0)) % 75264)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x1) + (7527*x0)) // 196) % 384), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x4), tmp38, xmask)
    tl.store(out_ptr1 + (x4), tmp39, xmask)
    tl.store(out_ptr2 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctjgs4b6jruwjfwtp7gtmfa35w2hc5y4va36ezz4vi4p37hrko45.py
# Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
# group_norm_27 => add_95, mul_121
triton_poi_fused_native_group_norm_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 75264.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlm4zd5ipdos6i5v6pf4gfmvrmbj5ygj2mzpzfwz5xvlju4rgtu.py
# Source Nodes: [mul_26, mul_27, sub_13, x_116, x_121, x_123], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
# mul_26 => mul_119
# mul_27 => mul_125
# sub_13 => sub_40
# x_116 => add_93
# x_121 => convolution_30
# x_123 => add_97
triton_poi_fused_add_convolution_mul_sub_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sub_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rsun4iwdqhjjybnat47xhtpn752xsghihh4emsfafabtdiov6c.py
# Source Nodes: [mul_58, mul_59, sub_29, x_244, x_252], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_58 => mul_263
# mul_59 => mul_269
# sub_29 => sub_88
# x_244 => add_205
# x_252 => add_209
triton_poi_fused_add_mul_sub_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfoqxevxht5npkd5f3b7vdtocwfv4fyqij2bx5hhxe3oi2ietok.py
# Source Nodes: [x_255], Original ATen: [aten.convolution]
# x_255 => convolution_63
triton_poi_fused_convolution_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5w6mkhfuuy72rbegvuti37gh2hqt5r37veky6z3plz3llquvlq.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_per_fused_native_group_norm_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp23 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp22, 0)
    tmp27 = tl.where(rmask & xmask, tmp23, 0)
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26, tmp27, tmp28, 1)
    tmp32 = tmp29[:, None]
    tmp33 = tmp30[:, None]
    tmp34 = tmp31[:, None]
    tl.store(out_ptr0 + (x5), tmp32, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
    tl.store(out_ptr2 + (x5), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vf5k6aurnxelouwfvmssfuud57fazu4shauy42wnv7zhbdaac2.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_per_fused_native_group_norm_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 59
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (59*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuj2g23dhd4vsmvsvkx4lc7gtdxnqcbzuyvpnrug6w66zjmpwcy.py
# Source Nodes: [group_norm_60], Original ATen: [aten.detach, aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_per_fused_detach_native_group_norm_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_detach_native_group_norm_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (5*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 37632.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnpog2r22qk6p4bhhrbu3t6mxzccvwb7cemt2dy6j2wgacl3477.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => add_211, mul_271
triton_poi_fused_native_group_norm_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 37632.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmy5jtghiscvyy4v3ge7cvadekmqesdcs55lkam6cewonh7e2eii.py
# Source Nodes: [y_30], Original ATen: [aten.avg_pool2d]
# y_30 => avg_pool2d_30
triton_poi_fused_avg_pool2d_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 7
    x1 = (xindex // 768) % 7
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6144) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-5376) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-4608) + x6), tmp27, other=0.0)
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
    tmp56 = tl.load(in_ptr0 + (4608 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (5376 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6144 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tl.store(out_ptr0 + (x6), tmp89, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jd7snzfq3lrgtb4ejl3bmg3knuza5y6exizooogx5mlxlqvep4.py
# Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
# group_norm_61 => var_mean_61
triton_per_fused_native_group_norm_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsxzf73pzzrfchpegs2kwcco6h3xwws3itr3xcaoo2ccslapabp.py
# Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
# group_norm_61 => add_214, mul_274
triton_poi_fused_native_group_norm_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 37632.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czji55fzr3ohzzahlce352oibfrgioukndtfiv4ymqfrkd7vqed4.py
# Source Nodes: [x_257], Original ATen: [aten.convolution]
# x_257 => convolution_64
triton_poi_fused_convolution_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3072
    y1 = (yindex // 3072)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (3072*x2) + (150528*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wlx73266oesmashudnnronmfk3dpionuams6wszyldjdmgio4d.py
# Source Nodes: [x_258], Original ATen: [aten.gelu]
# x_258 => add_215, erf_30, mul_275, mul_276, mul_277
triton_poi_fused_gelu_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6rnb5eiacgkj7q6r3nzwonxc7cp53qtlmescl3u5q4ws75djx7.py
# Source Nodes: [mul_60, mul_61, sub_30, x_256, x_263], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_60 => mul_272
# mul_61 => mul_278
# sub_30 => sub_91
# x_256 => add_212
# x_263 => add_216
triton_poi_fused_add_mul_sub_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (y0 + (49*x2) + (37632*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl65felxztel4c7eabwpolj64t5wlpyaxf3q7etxjk425hpzmszm.py
# Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
# group_norm_62 => var_mean_62
triton_red_fused_native_group_norm_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgvhvklyqfcp3o66irlrvxctsb5tguv7f2n67xv6nl6eqs5ncso.py
# Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
# group_norm_62 => add_218, mul_280
triton_poi_fused_native_group_norm_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 37632.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosuxnonfkylvvfzwsxjxpl2hsaxfzx2vqwxfkytedeytrjxpczh.py
# Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
# group_norm_63 => var_mean_63
triton_per_fused_native_group_norm_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((37632*x2) + ((r3 + (128*x0) + (7527*x1)) % 37632)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyn7v4cl7zhlqvux7swa7226qmhz43zygz75volalvuydyiun3k.py
# Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
# group_norm_63 => add_221, mul_283
triton_poi_fused_native_group_norm_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 37632.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcanefwv7dq6idniz3vln7ycauqccbrrx4g4tozvw4fesc3aurv.py
# Source Nodes: [mul_62, mul_63, sub_31, x_264, x_269, x_271], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
# mul_62 => mul_281
# mul_63 => mul_287
# sub_31 => sub_94
# x_264 => add_219
# x_269 => convolution_67
# x_271 => add_223
triton_poi_fused_add_convolution_mul_sub_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sub_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hub3tfjcia2qdjfmcvnyhgxgj2igubckrfk3pebmxnxeojmhsu.py
# Source Nodes: [mul_70, mul_71, sub_35, x_296, x_301, x_306, x_307], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.sub]
# mul_70 => mul_317
# mul_71 => mul_323
# sub_35 => sub_106
# x_296 => add_247
# x_301 => convolution_75
# x_306 => add_251
# x_307 => mean
triton_per_fused_add_convolution_mean_mul_sub_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_mean_mul_sub_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr4 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0 + (768*r2) + (37632*x1)), tmp2, rmask)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coe66uncg3hnjzfobc6qv6psqmdnrd2zlqoon5idobmoe7kt26f3.py
# Source Nodes: [x_311, x_314], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_311 => add_252, mul_324, rsqrt_72, sub_108, var_mean_72
# x_314 => view_216
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373 = args
    args.clear()
    assert_size_stride(primals_1, (96, ), (1, ))
    assert_size_stride(primals_2, (96, ), (1, ))
    assert_size_stride(primals_3, (96, ), (1, ))
    assert_size_stride(primals_4, (96, ), (1, ))
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
    assert_size_stride(primals_15, (96, ), (1, ))
    assert_size_stride(primals_16, (96, ), (1, ))
    assert_size_stride(primals_17, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (96, ), (1, ))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_26, (96, ), (1, ))
    assert_size_stride(primals_27, (96, ), (1, ))
    assert_size_stride(primals_28, (96, ), (1, ))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (192, ), (1, ))
    assert_size_stride(primals_63, (192, ), (1, ))
    assert_size_stride(primals_64, (192, ), (1, ))
    assert_size_stride(primals_65, (192, ), (1, ))
    assert_size_stride(primals_66, (192, ), (1, ))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_68, (192, ), (1, ))
    assert_size_stride(primals_69, (192, ), (1, ))
    assert_size_stride(primals_70, (192, ), (1, ))
    assert_size_stride(primals_71, (192, ), (1, ))
    assert_size_stride(primals_72, (192, ), (1, ))
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
    assert_size_stride(primals_119, (384, ), (1, ))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, ), (1, ))
    assert_size_stride(primals_122, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (384, ), (1, ))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (384, ), (1, ))
    assert_size_stride(primals_128, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (384, ), (1, ))
    assert_size_stride(primals_132, (384, ), (1, ))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (384, ), (1, ))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (384, ), (1, ))
    assert_size_stride(primals_144, (384, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, ), (1, ))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, ), (1, ))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (96, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_220, (96, ), (1, ))
    assert_size_stride(primals_221, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_224, (96, ), (1, ))
    assert_size_stride(primals_225, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_228, (96, ), (1, ))
    assert_size_stride(primals_229, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_232, (96, ), (1, ))
    assert_size_stride(primals_233, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_236, (96, ), (1, ))
    assert_size_stride(primals_237, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_240, (96, ), (1, ))
    assert_size_stride(primals_241, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_244, (96, ), (1, ))
    assert_size_stride(primals_245, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_246, (192, ), (1, ))
    assert_size_stride(primals_247, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (192, ), (1, ))
    assert_size_stride(primals_251, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_254, (192, ), (1, ))
    assert_size_stride(primals_255, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_256, (768, ), (1, ))
    assert_size_stride(primals_257, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_258, (192, ), (1, ))
    assert_size_stride(primals_259, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_260, (768, ), (1, ))
    assert_size_stride(primals_261, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_262, (192, ), (1, ))
    assert_size_stride(primals_263, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_264, (768, ), (1, ))
    assert_size_stride(primals_265, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_266, (192, ), (1, ))
    assert_size_stride(primals_267, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_268, (768, ), (1, ))
    assert_size_stride(primals_269, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_270, (192, ), (1, ))
    assert_size_stride(primals_271, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_272, (384, ), (1, ))
    assert_size_stride(primals_273, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_274, (1536, ), (1, ))
    assert_size_stride(primals_275, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_276, (384, ), (1, ))
    assert_size_stride(primals_277, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_278, (1536, ), (1, ))
    assert_size_stride(primals_279, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_282, (1536, ), (1, ))
    assert_size_stride(primals_283, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_284, (384, ), (1, ))
    assert_size_stride(primals_285, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_286, (1536, ), (1, ))
    assert_size_stride(primals_287, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_288, (384, ), (1, ))
    assert_size_stride(primals_289, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_290, (1536, ), (1, ))
    assert_size_stride(primals_291, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_292, (384, ), (1, ))
    assert_size_stride(primals_293, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_294, (1536, ), (1, ))
    assert_size_stride(primals_295, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_296, (384, ), (1, ))
    assert_size_stride(primals_297, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_298, (1536, ), (1, ))
    assert_size_stride(primals_299, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_300, (384, ), (1, ))
    assert_size_stride(primals_301, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_302, (1536, ), (1, ))
    assert_size_stride(primals_303, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_304, (384, ), (1, ))
    assert_size_stride(primals_305, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_306, (1536, ), (1, ))
    assert_size_stride(primals_307, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_308, (384, ), (1, ))
    assert_size_stride(primals_309, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_310, (1536, ), (1, ))
    assert_size_stride(primals_311, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_312, (384, ), (1, ))
    assert_size_stride(primals_313, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_314, (1536, ), (1, ))
    assert_size_stride(primals_315, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_316, (384, ), (1, ))
    assert_size_stride(primals_317, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_318, (1536, ), (1, ))
    assert_size_stride(primals_319, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_320, (384, ), (1, ))
    assert_size_stride(primals_321, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_322, (1536, ), (1, ))
    assert_size_stride(primals_323, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_324, (384, ), (1, ))
    assert_size_stride(primals_325, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_326, (1536, ), (1, ))
    assert_size_stride(primals_327, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_328, (384, ), (1, ))
    assert_size_stride(primals_329, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_330, (1536, ), (1, ))
    assert_size_stride(primals_331, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_332, (384, ), (1, ))
    assert_size_stride(primals_333, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_334, (1536, ), (1, ))
    assert_size_stride(primals_335, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_336, (384, ), (1, ))
    assert_size_stride(primals_337, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_338, (1536, ), (1, ))
    assert_size_stride(primals_339, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_340, (384, ), (1, ))
    assert_size_stride(primals_341, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_342, (1536, ), (1, ))
    assert_size_stride(primals_343, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_344, (384, ), (1, ))
    assert_size_stride(primals_345, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_346, (768, ), (1, ))
    assert_size_stride(primals_347, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_348, (3072, ), (1, ))
    assert_size_stride(primals_349, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_350, (768, ), (1, ))
    assert_size_stride(primals_351, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_352, (3072, ), (1, ))
    assert_size_stride(primals_353, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_354, (768, ), (1, ))
    assert_size_stride(primals_355, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_356, (3072, ), (1, ))
    assert_size_stride(primals_357, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_358, (768, ), (1, ))
    assert_size_stride(primals_359, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_360, (3072, ), (1, ))
    assert_size_stride(primals_361, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_364, (3072, ), (1, ))
    assert_size_stride(primals_365, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_368, (3072, ), (1, ))
    assert_size_stride(primals_369, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_370, (768, ), (1, ))
    assert_size_stride(primals_371, (1000, 768), (768, 1))
    assert_size_stride(primals_372, (1000, ), (1, ))
    assert_size_stride(primals_373, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((96, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_219, buf0, 288, 49, grid=grid(288, 49), stream=stream0)
        del primals_219
        buf1 = empty_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_245, buf1, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del primals_245
        buf2 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_271, buf2, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_271
        buf3 = empty_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_345, buf3, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_345
        buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_373, buf4, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_373
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, buf0, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf6 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf5, primals_220, buf6, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_220
        buf7 = empty_strided((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_6.run(buf6, buf7, buf8, buf9, 18944, 128, grid=grid(18944), stream=stream0)
        buf10 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf7, buf8, buf9, buf10, buf11, buf12, 296, 64, grid=grid(296), stream=stream0)
        buf13 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf967 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf10, buf11, buf12, buf13, buf14, buf967, 8, 37, grid=grid(8), stream=stream0)
        buf16 = reinterpret_tensor(buf5, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf5  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_9.run(buf6, buf13, buf14, primals_1, primals_2, buf16, 2408448, grid=grid(2408448), stream=stream0)
        del primals_2
        buf17 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf16, buf17, 2408448, grid=grid(2408448), stream=stream0)
        buf18 = buf9; del buf9  # reuse
        buf19 = buf8; del buf8  # reuse
        buf20 = buf7; del buf7  # reuse
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_11.run(buf6, buf17, buf16, primals_3, buf18, buf19, buf20, 18944, 128, grid=grid(18944), stream=stream0)
        buf21 = buf12; del buf12  # reuse
        buf22 = buf11; del buf11  # reuse
        buf23 = buf10; del buf10  # reuse
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf18, buf19, buf20, buf21, buf22, buf23, 296, 64, grid=grid(296), stream=stream0)
        buf24 = buf14; del buf14  # reuse
        buf25 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf966 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_1], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf21, buf22, buf23, buf24, buf25, buf966, 8, 37, grid=grid(8), stream=stream0)
        buf27 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_12.run(buf6, buf17, buf16, primals_3, buf24, buf25, primals_4, primals_5, buf27, 2408448, grid=grid(2408448), stream=stream0)
        del primals_5
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf29 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf28, primals_222, buf29, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_222
        buf30 = reinterpret_tensor(buf28, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf28  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf29, buf30, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf32 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf31, primals_224, buf32, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_224
        buf33 = buf31; del buf31  # reuse
        # Source Nodes: [mul, mul_1, sub, x_11, x_4], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_15.run(buf6, buf17, buf16, primals_3, buf32, primals_6, buf33, 25088, 96, grid=grid(25088, 96), stream=stream0)
        buf34 = buf23; del buf23  # reuse
        buf35 = buf22; del buf22  # reuse
        buf36 = buf21; del buf21  # reuse
        # Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_16.run(buf33, buf34, buf35, buf36, 296, 8137, grid=grid(296), stream=stream0)
        buf37 = buf25; del buf25  # reuse
        buf38 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf965 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_2], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf34, buf35, buf36, buf37, buf38, buf965, 8, 37, grid=grid(8), stream=stream0)
        buf40 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf33, buf37, buf38, primals_7, primals_8, buf40, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_8
        buf41 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_1], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf40, buf41, 2408448, grid=grid(2408448), stream=stream0)
        buf42 = buf20; del buf20  # reuse
        buf43 = buf19; del buf19  # reuse
        buf44 = buf18; del buf18  # reuse
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf33, buf41, buf40, primals_9, buf42, buf43, buf44, 18944, 128, grid=grid(18944), stream=stream0)
        buf45 = buf36; del buf36  # reuse
        buf46 = buf35; del buf35  # reuse
        buf47 = buf34; del buf34  # reuse
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf42, buf43, buf44, buf45, buf46, buf47, 296, 64, grid=grid(296), stream=stream0)
        buf48 = buf38; del buf38  # reuse
        buf49 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf964 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_3], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf45, buf46, buf47, buf48, buf49, buf964, 8, 37, grid=grid(8), stream=stream0)
        buf51 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_19.run(buf33, buf41, buf40, primals_9, buf48, buf49, primals_10, primals_11, buf51, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_11
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf53 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf52, primals_226, buf53, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_226
        buf54 = reinterpret_tensor(buf52, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf52  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf53, buf54, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf56 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        buf57 = buf33; del buf33  # reuse
        # Source Nodes: [mul_2, mul_3, sub_1, x_12, x_17, x_19], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_20.run(buf57, buf55, primals_228, buf41, buf40, primals_9, primals_12, buf56, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_228
        buf58 = buf47; del buf47  # reuse
        buf59 = buf46; del buf46  # reuse
        buf60 = buf45; del buf45  # reuse
        # Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_16.run(buf57, buf58, buf59, buf60, 296, 8137, grid=grid(296), stream=stream0)
        buf61 = buf49; del buf49  # reuse
        buf62 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf963 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_4], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf58, buf59, buf60, buf61, buf62, buf963, 8, 37, grid=grid(8), stream=stream0)
        buf64 = reinterpret_tensor(buf55, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf55  # reuse
        # Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf57, buf61, buf62, primals_13, primals_14, buf64, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_14
        buf65 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_2], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf64, buf65, 2408448, grid=grid(2408448), stream=stream0)
        buf66 = buf44; del buf44  # reuse
        buf67 = buf43; del buf43  # reuse
        buf68 = buf42; del buf42  # reuse
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf57, buf65, buf64, primals_15, buf66, buf67, buf68, 18944, 128, grid=grid(18944), stream=stream0)
        buf69 = buf60; del buf60  # reuse
        buf70 = buf59; del buf59  # reuse
        buf71 = buf58; del buf58  # reuse
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf66, buf67, buf68, buf69, buf70, buf71, 296, 64, grid=grid(296), stream=stream0)
        buf72 = buf62; del buf62  # reuse
        buf73 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf962 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_5], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf69, buf70, buf71, buf72, buf73, buf962, 8, 37, grid=grid(8), stream=stream0)
        buf75 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_19.run(buf57, buf65, buf64, primals_15, buf72, buf73, primals_16, primals_17, buf75, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_17
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf77 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf76, primals_230, buf77, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_230
        buf78 = reinterpret_tensor(buf76, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf76  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf77, buf78, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf80 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        buf81 = buf57; del buf57  # reuse
        # Source Nodes: [mul_4, mul_5, sub_2, x_20, x_25, x_27], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_20.run(buf81, buf79, primals_232, buf65, buf64, primals_15, primals_18, buf80, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_232
        buf82 = buf71; del buf71  # reuse
        buf83 = buf70; del buf70  # reuse
        buf84 = buf69; del buf69  # reuse
        # Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_16.run(buf81, buf82, buf83, buf84, 296, 8137, grid=grid(296), stream=stream0)
        buf85 = buf73; del buf73  # reuse
        buf86 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf961 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_6], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf82, buf83, buf84, buf85, buf86, buf961, 8, 37, grid=grid(8), stream=stream0)
        buf88 = reinterpret_tensor(buf79, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf79  # reuse
        # Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf81, buf85, buf86, primals_19, primals_20, buf88, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_20
        buf89 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_3], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf88, buf89, 2408448, grid=grid(2408448), stream=stream0)
        buf90 = buf68; del buf68  # reuse
        buf91 = buf67; del buf67  # reuse
        buf92 = buf66; del buf66  # reuse
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf81, buf89, buf88, primals_21, buf90, buf91, buf92, 18944, 128, grid=grid(18944), stream=stream0)
        buf93 = buf84; del buf84  # reuse
        buf94 = buf83; del buf83  # reuse
        buf95 = buf82; del buf82  # reuse
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf90, buf91, buf92, buf93, buf94, buf95, 296, 64, grid=grid(296), stream=stream0)
        buf96 = buf86; del buf86  # reuse
        buf97 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf960 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_7], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf93, buf94, buf95, buf96, buf97, buf960, 8, 37, grid=grid(8), stream=stream0)
        buf99 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_19.run(buf81, buf89, buf88, primals_21, buf96, buf97, primals_22, primals_23, buf99, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_23
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf101 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf100, primals_234, buf101, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_234
        buf102 = reinterpret_tensor(buf100, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf100  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf101, buf102, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf104 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        buf105 = buf81; del buf81  # reuse
        # Source Nodes: [mul_6, mul_7, sub_3, x_28, x_33, x_35], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_20.run(buf105, buf103, primals_236, buf89, buf88, primals_21, primals_24, buf104, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_236
        buf106 = buf95; del buf95  # reuse
        buf107 = buf94; del buf94  # reuse
        buf108 = buf93; del buf93  # reuse
        # Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_16.run(buf105, buf106, buf107, buf108, 296, 8137, grid=grid(296), stream=stream0)
        buf109 = buf97; del buf97  # reuse
        buf110 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf959 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_8], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf106, buf107, buf108, buf109, buf110, buf959, 8, 37, grid=grid(8), stream=stream0)
        buf112 = reinterpret_tensor(buf103, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf103  # reuse
        # Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf105, buf109, buf110, primals_25, primals_26, buf112, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_26
        buf113 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf112, buf113, 2408448, grid=grid(2408448), stream=stream0)
        buf114 = buf92; del buf92  # reuse
        buf115 = buf91; del buf91  # reuse
        buf116 = buf90; del buf90  # reuse
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf105, buf113, buf112, primals_27, buf114, buf115, buf116, 18944, 128, grid=grid(18944), stream=stream0)
        buf117 = buf108; del buf108  # reuse
        buf118 = buf107; del buf107  # reuse
        buf119 = buf106; del buf106  # reuse
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf114, buf115, buf116, buf117, buf118, buf119, 296, 64, grid=grid(296), stream=stream0)
        buf120 = buf110; del buf110  # reuse
        buf121 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf958 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_9], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf117, buf118, buf119, buf120, buf121, buf958, 8, 37, grid=grid(8), stream=stream0)
        buf123 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_19.run(buf105, buf113, buf112, primals_27, buf120, buf121, primals_28, primals_29, buf123, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_29
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf125 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf124, primals_238, buf125, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_238
        buf126 = reinterpret_tensor(buf124, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf124  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf125, buf126, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_239, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf128 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        buf129 = buf105; del buf105  # reuse
        # Source Nodes: [mul_8, mul_9, sub_4, x_36, x_41, x_43], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_20.run(buf129, buf127, primals_240, buf113, buf112, primals_27, primals_30, buf128, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_240
        buf130 = buf119; del buf119  # reuse
        buf131 = buf118; del buf118  # reuse
        buf132 = buf117; del buf117  # reuse
        # Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_16.run(buf129, buf130, buf131, buf132, 296, 8137, grid=grid(296), stream=stream0)
        buf133 = buf121; del buf121  # reuse
        buf134 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf957 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_10], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf130, buf131, buf132, buf133, buf134, buf957, 8, 37, grid=grid(8), stream=stream0)
        buf136 = reinterpret_tensor(buf127, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf127  # reuse
        # Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf129, buf133, buf134, primals_31, primals_32, buf136, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_32
        buf137 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf136, buf137, 2408448, grid=grid(2408448), stream=stream0)
        buf138 = buf116; del buf116  # reuse
        buf139 = buf115; del buf115  # reuse
        buf140 = buf114; del buf114  # reuse
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf129, buf137, buf136, primals_33, buf138, buf139, buf140, 18944, 128, grid=grid(18944), stream=stream0)
        buf141 = buf132; del buf132  # reuse
        buf142 = buf131; del buf131  # reuse
        buf143 = buf130; del buf130  # reuse
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_7.run(buf138, buf139, buf140, buf141, buf142, buf143, 296, 64, grid=grid(296), stream=stream0)
        del buf138
        del buf139
        del buf140
        buf144 = buf134; del buf134  # reuse
        buf145 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf956 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_11], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_8.run(buf141, buf142, buf143, buf144, buf145, buf956, 8, 37, grid=grid(8), stream=stream0)
        del buf141
        del buf142
        del buf143
        buf147 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_19.run(buf129, buf137, buf136, primals_33, buf144, buf145, primals_34, primals_35, buf147, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_35
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf149 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf148, primals_242, buf149, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del primals_242
        buf150 = reinterpret_tensor(buf148, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf148  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf149, buf150, 9633792, grid=grid(9633792), stream=stream0)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf152 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf151, primals_244, buf152, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_244
        buf153 = reinterpret_tensor(buf151, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf151  # reuse
        # Source Nodes: [mul_10, mul_11, sub_5, x_44, x_52], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_21.run(buf129, buf137, buf136, primals_33, buf152, primals_36, buf153, 25088, 96, grid=grid(25088, 96), stream=stream0)
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf155 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf154, primals_246, buf155, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_246
        buf156 = empty_strided((8, 1, 1, 1, 19, 62), (1178, 9424, 9424, 9424, 1, 19), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((8, 1, 1, 1, 19, 62), (1178, 9424, 9424, 9424, 1, 19), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((8, 1, 1, 1, 19, 62), (1178, 9424, 9424, 9424, 1, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf155, buf156, buf157, buf158, 9424, 128, grid=grid(9424), stream=stream0)
        buf159 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        buf160 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        buf161 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf156, buf157, buf158, buf159, buf160, buf161, 152, 62, grid=grid(152), stream=stream0)
        buf162 = buf145; del buf145  # reuse
        buf163 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf955 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_12], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf159, buf160, buf161, buf162, buf163, buf955, 8, 19, grid=grid(8), stream=stream0)
        buf165 = reinterpret_tensor(buf154, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf154  # reuse
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_26.run(buf155, buf162, buf163, primals_37, primals_38, buf165, 1204224, grid=grid(1204224), stream=stream0)
        del primals_38
        buf166 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_6], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf165, buf166, 1204224, grid=grid(1204224), stream=stream0)
        buf167 = buf158; del buf158  # reuse
        buf168 = buf157; del buf157  # reuse
        buf169 = buf156; del buf156  # reuse
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf155, buf166, buf165, primals_39, buf167, buf168, buf169, 9424, 128, grid=grid(9424), stream=stream0)
        buf170 = buf161; del buf161  # reuse
        buf171 = buf160; del buf160  # reuse
        buf172 = buf159; del buf159  # reuse
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf167, buf168, buf169, buf170, buf171, buf172, 152, 62, grid=grid(152), stream=stream0)
        buf173 = buf163; del buf163  # reuse
        buf174 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf954 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_13], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf170, buf171, buf172, buf173, buf174, buf954, 8, 19, grid=grid(8), stream=stream0)
        buf176 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_29.run(buf155, buf166, buf165, primals_39, buf173, buf174, primals_40, primals_41, buf176, 1204224, grid=grid(1204224), stream=stream0)
        del primals_41
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf178 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf177, primals_248, buf178, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_248
        buf179 = reinterpret_tensor(buf177, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf177  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf178, buf179, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf181 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf180, primals_250, buf181, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_250
        buf182 = buf180; del buf180  # reuse
        # Source Nodes: [mul_12, mul_13, sub_6, x_56, x_63], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_32.run(buf155, buf166, buf165, primals_39, buf181, primals_42, buf182, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf183 = buf172; del buf172  # reuse
        buf184 = buf171; del buf171  # reuse
        buf185 = buf170; del buf170  # reuse
        # Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_33.run(buf182, buf183, buf184, buf185, 152, 7923, grid=grid(152), stream=stream0)
        buf186 = buf174; del buf174  # reuse
        buf187 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf953 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_14], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf183, buf184, buf185, buf186, buf187, buf953, 8, 19, grid=grid(8), stream=stream0)
        buf189 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_34.run(buf182, buf186, buf187, primals_43, primals_44, buf189, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_44
        buf190 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf189, buf190, 1204224, grid=grid(1204224), stream=stream0)
        buf191 = buf169; del buf169  # reuse
        buf192 = buf168; del buf168  # reuse
        buf193 = buf167; del buf167  # reuse
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_35.run(buf182, buf190, buf189, primals_45, buf191, buf192, buf193, 9424, 128, grid=grid(9424), stream=stream0)
        buf194 = buf185; del buf185  # reuse
        buf195 = buf184; del buf184  # reuse
        buf196 = buf183; del buf183  # reuse
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf191, buf192, buf193, buf194, buf195, buf196, 152, 62, grid=grid(152), stream=stream0)
        buf197 = buf187; del buf187  # reuse
        buf198 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf952 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_15], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf194, buf195, buf196, buf197, buf198, buf952, 8, 19, grid=grid(8), stream=stream0)
        buf200 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_36.run(buf182, buf190, buf189, primals_45, buf197, buf198, primals_46, primals_47, buf200, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_47
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf202 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf201, primals_252, buf202, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_252
        buf203 = reinterpret_tensor(buf201, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf201  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf202, buf203, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf205 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf206 = buf182; del buf182  # reuse
        # Source Nodes: [mul_14, mul_15, sub_7, x_64, x_69, x_71], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_37.run(buf206, buf204, primals_254, buf190, buf189, primals_45, primals_48, buf205, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_254
        buf207 = buf196; del buf196  # reuse
        buf208 = buf195; del buf195  # reuse
        buf209 = buf194; del buf194  # reuse
        # Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_33.run(buf206, buf207, buf208, buf209, 152, 7923, grid=grid(152), stream=stream0)
        buf210 = buf198; del buf198  # reuse
        buf211 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf951 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_16], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf207, buf208, buf209, buf210, buf211, buf951, 8, 19, grid=grid(8), stream=stream0)
        buf213 = reinterpret_tensor(buf204, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf204  # reuse
        # Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_34.run(buf206, buf210, buf211, primals_49, primals_50, buf213, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_50
        buf214 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_8], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf213, buf214, 1204224, grid=grid(1204224), stream=stream0)
        buf215 = buf193; del buf193  # reuse
        buf216 = buf192; del buf192  # reuse
        buf217 = buf191; del buf191  # reuse
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_35.run(buf206, buf214, buf213, primals_51, buf215, buf216, buf217, 9424, 128, grid=grid(9424), stream=stream0)
        buf218 = buf209; del buf209  # reuse
        buf219 = buf208; del buf208  # reuse
        buf220 = buf207; del buf207  # reuse
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf215, buf216, buf217, buf218, buf219, buf220, 152, 62, grid=grid(152), stream=stream0)
        buf221 = buf211; del buf211  # reuse
        buf222 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf950 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_17], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf218, buf219, buf220, buf221, buf222, buf950, 8, 19, grid=grid(8), stream=stream0)
        buf224 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_36.run(buf206, buf214, buf213, primals_51, buf221, buf222, primals_52, primals_53, buf224, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_53
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf226 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf225, primals_256, buf226, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_256
        buf227 = reinterpret_tensor(buf225, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf225  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf226, buf227, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf229 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf230 = buf206; del buf206  # reuse
        # Source Nodes: [mul_16, mul_17, sub_8, x_72, x_77, x_79], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_37.run(buf230, buf228, primals_258, buf214, buf213, primals_51, primals_54, buf229, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_258
        buf231 = buf220; del buf220  # reuse
        buf232 = buf219; del buf219  # reuse
        buf233 = buf218; del buf218  # reuse
        # Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_33.run(buf230, buf231, buf232, buf233, 152, 7923, grid=grid(152), stream=stream0)
        buf234 = buf222; del buf222  # reuse
        buf235 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf949 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_18], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf231, buf232, buf233, buf234, buf235, buf949, 8, 19, grid=grid(8), stream=stream0)
        buf237 = reinterpret_tensor(buf228, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf228  # reuse
        # Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_34.run(buf230, buf234, buf235, primals_55, primals_56, buf237, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_56
        buf238 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_9], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf237, buf238, 1204224, grid=grid(1204224), stream=stream0)
        buf239 = buf217; del buf217  # reuse
        buf240 = buf216; del buf216  # reuse
        buf241 = buf215; del buf215  # reuse
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_35.run(buf230, buf238, buf237, primals_57, buf239, buf240, buf241, 9424, 128, grid=grid(9424), stream=stream0)
        buf242 = buf233; del buf233  # reuse
        buf243 = buf232; del buf232  # reuse
        buf244 = buf231; del buf231  # reuse
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf239, buf240, buf241, buf242, buf243, buf244, 152, 62, grid=grid(152), stream=stream0)
        buf245 = buf235; del buf235  # reuse
        buf246 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf948 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_19], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf242, buf243, buf244, buf245, buf246, buf948, 8, 19, grid=grid(8), stream=stream0)
        buf248 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_36.run(buf230, buf238, buf237, primals_57, buf245, buf246, primals_58, primals_59, buf248, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_59
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf250 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf249, primals_260, buf250, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_260
        buf251 = reinterpret_tensor(buf249, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf249  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf250, buf251, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf253 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf254 = buf230; del buf230  # reuse
        # Source Nodes: [mul_18, mul_19, sub_9, x_80, x_85, x_87], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_37.run(buf254, buf252, primals_262, buf238, buf237, primals_57, primals_60, buf253, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_262
        buf255 = buf244; del buf244  # reuse
        buf256 = buf243; del buf243  # reuse
        buf257 = buf242; del buf242  # reuse
        # Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_33.run(buf254, buf255, buf256, buf257, 152, 7923, grid=grid(152), stream=stream0)
        buf258 = buf246; del buf246  # reuse
        buf259 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf947 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_20], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf255, buf256, buf257, buf258, buf259, buf947, 8, 19, grid=grid(8), stream=stream0)
        buf261 = reinterpret_tensor(buf252, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf252  # reuse
        # Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_34.run(buf254, buf258, buf259, primals_61, primals_62, buf261, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_62
        buf262 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_10], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf261, buf262, 1204224, grid=grid(1204224), stream=stream0)
        buf263 = buf241; del buf241  # reuse
        buf264 = buf240; del buf240  # reuse
        buf265 = buf239; del buf239  # reuse
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_35.run(buf254, buf262, buf261, primals_63, buf263, buf264, buf265, 9424, 128, grid=grid(9424), stream=stream0)
        buf266 = buf257; del buf257  # reuse
        buf267 = buf256; del buf256  # reuse
        buf268 = buf255; del buf255  # reuse
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf263, buf264, buf265, buf266, buf267, buf268, 152, 62, grid=grid(152), stream=stream0)
        buf269 = buf259; del buf259  # reuse
        buf270 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf946 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_21], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf266, buf267, buf268, buf269, buf270, buf946, 8, 19, grid=grid(8), stream=stream0)
        buf272 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_36.run(buf254, buf262, buf261, primals_63, buf269, buf270, primals_64, primals_65, buf272, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_65
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf274 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf273, primals_264, buf274, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_264
        buf275 = reinterpret_tensor(buf273, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf273  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf274, buf275, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf277 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf278 = buf254; del buf254  # reuse
        # Source Nodes: [mul_20, mul_21, sub_10, x_88, x_93, x_95], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_37.run(buf278, buf276, primals_266, buf262, buf261, primals_63, primals_66, buf277, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_266
        buf279 = buf268; del buf268  # reuse
        buf280 = buf267; del buf267  # reuse
        buf281 = buf266; del buf266  # reuse
        # Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_33.run(buf278, buf279, buf280, buf281, 152, 7923, grid=grid(152), stream=stream0)
        buf282 = buf270; del buf270  # reuse
        buf283 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf945 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_22], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf279, buf280, buf281, buf282, buf283, buf945, 8, 19, grid=grid(8), stream=stream0)
        buf285 = reinterpret_tensor(buf276, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf276  # reuse
        # Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_34.run(buf278, buf282, buf283, primals_67, primals_68, buf285, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_68
        buf286 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_11], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_27.run(buf285, buf286, 1204224, grid=grid(1204224), stream=stream0)
        buf287 = buf265; del buf265  # reuse
        buf288 = buf264; del buf264  # reuse
        buf289 = buf263; del buf263  # reuse
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_35.run(buf278, buf286, buf285, primals_69, buf287, buf288, buf289, 9424, 128, grid=grid(9424), stream=stream0)
        buf290 = buf281; del buf281  # reuse
        buf291 = buf280; del buf280  # reuse
        buf292 = buf279; del buf279  # reuse
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf287, buf288, buf289, buf290, buf291, buf292, 152, 62, grid=grid(152), stream=stream0)
        del buf287
        del buf288
        del buf289
        buf293 = buf283; del buf283  # reuse
        buf294 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf944 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_23], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_25.run(buf290, buf291, buf292, buf293, buf294, buf944, 8, 19, grid=grid(8), stream=stream0)
        del buf290
        del buf291
        del buf292
        buf296 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_36.run(buf278, buf286, buf285, primals_69, buf293, buf294, primals_70, primals_71, buf296, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_71
        # Source Nodes: [x_97], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf298 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf297, primals_268, buf298, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_268
        buf299 = reinterpret_tensor(buf297, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf297  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf298, buf299, 4816896, grid=grid(4816896), stream=stream0)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf301 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf300, primals_270, buf301, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_270
        buf302 = reinterpret_tensor(buf300, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf300  # reuse
        # Source Nodes: [mul_22, mul_23, sub_11, x_104, x_96], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_38.run(buf278, buf286, buf285, primals_69, buf301, primals_72, buf302, 6272, 192, grid=grid(6272, 192), stream=stream0)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf304 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf303, primals_272, buf304, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_272
        buf305 = empty_strided((8, 1, 1, 1, 10, 59), (590, 4720, 4720, 4720, 1, 10), device='cuda', dtype=torch.float32)
        buf306 = empty_strided((8, 1, 1, 1, 10, 59), (590, 4720, 4720, 4720, 1, 10), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((8, 1, 1, 1, 10, 59), (590, 4720, 4720, 4720, 1, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf304, buf305, buf306, buf307, 4720, 128, grid=grid(4720), stream=stream0)
        buf308 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        buf309 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf305, buf306, buf307, buf308, buf309, buf310, 80, 59, grid=grid(80), stream=stream0)
        buf311 = buf294; del buf294  # reuse
        buf312 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf943 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_24], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf308, buf309, buf310, buf311, buf312, buf943, 8, 10, grid=grid(8), stream=stream0)
        buf314 = reinterpret_tensor(buf303, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf303  # reuse
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_43.run(buf304, buf311, buf312, primals_73, primals_74, buf314, 602112, grid=grid(602112), stream=stream0)
        del primals_74
        buf315 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_12], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf314, buf315, 602112, grid=grid(602112), stream=stream0)
        buf316 = buf307; del buf307  # reuse
        buf317 = buf306; del buf306  # reuse
        buf318 = buf305; del buf305  # reuse
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf304, buf315, buf314, primals_75, buf316, buf317, buf318, 4720, 128, grid=grid(4720), stream=stream0)
        buf319 = buf310; del buf310  # reuse
        buf320 = buf309; del buf309  # reuse
        buf321 = buf308; del buf308  # reuse
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf316, buf317, buf318, buf319, buf320, buf321, 80, 59, grid=grid(80), stream=stream0)
        buf322 = buf312; del buf312  # reuse
        buf323 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf942 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_25], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf319, buf320, buf321, buf322, buf323, buf942, 8, 10, grid=grid(8), stream=stream0)
        buf325 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_46.run(buf304, buf315, buf314, primals_75, buf322, buf323, primals_76, primals_77, buf325, 602112, grid=grid(602112), stream=stream0)
        del primals_77
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf327 = reinterpret_tensor(buf129, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf129  # reuse
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf326, primals_274, buf327, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_274
        buf328 = reinterpret_tensor(buf326, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf326  # reuse
        # Source Nodes: [x_110], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf327, buf328, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf330 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf329, primals_276, buf330, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_276
        buf331 = buf329; del buf329  # reuse
        # Source Nodes: [mul_24, mul_25, sub_12, x_108, x_115], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_49.run(buf304, buf315, buf314, primals_75, buf330, primals_78, buf331, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf332 = buf321; del buf321  # reuse
        buf333 = buf320; del buf320  # reuse
        buf334 = buf319; del buf319  # reuse
        # Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf331, buf332, buf333, buf334, 80, 7527, grid=grid(80), stream=stream0)
        buf335 = buf323; del buf323  # reuse
        buf336 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf941 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_26], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf332, buf333, buf334, buf335, buf336, buf941, 8, 10, grid=grid(8), stream=stream0)
        buf338 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf331, buf335, buf336, primals_79, primals_80, buf338, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_80
        buf339 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_13], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf338, buf339, 602112, grid=grid(602112), stream=stream0)
        buf340 = buf318; del buf318  # reuse
        buf341 = buf317; del buf317  # reuse
        buf342 = buf316; del buf316  # reuse
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf331, buf339, buf338, primals_81, buf340, buf341, buf342, 4720, 128, grid=grid(4720), stream=stream0)
        buf343 = buf334; del buf334  # reuse
        buf344 = buf333; del buf333  # reuse
        buf345 = buf332; del buf332  # reuse
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf340, buf341, buf342, buf343, buf344, buf345, 80, 59, grid=grid(80), stream=stream0)
        buf346 = buf336; del buf336  # reuse
        buf347 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf940 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_27], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf343, buf344, buf345, buf346, buf347, buf940, 8, 10, grid=grid(8), stream=stream0)
        buf349 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf331, buf339, buf338, primals_81, buf346, buf347, primals_82, primals_83, buf349, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_83
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf351 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf350, primals_278, buf351, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_278
        buf352 = reinterpret_tensor(buf350, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf350  # reuse
        # Source Nodes: [x_118], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf351, buf352, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf354 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf355 = buf331; del buf331  # reuse
        # Source Nodes: [mul_26, mul_27, sub_13, x_116, x_121, x_123], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf355, buf353, primals_280, buf339, buf338, primals_81, primals_84, buf354, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_280
        buf356 = buf345; del buf345  # reuse
        buf357 = buf344; del buf344  # reuse
        buf358 = buf343; del buf343  # reuse
        # Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf355, buf356, buf357, buf358, 80, 7527, grid=grid(80), stream=stream0)
        buf359 = buf347; del buf347  # reuse
        buf360 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf939 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_28], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf356, buf357, buf358, buf359, buf360, buf939, 8, 10, grid=grid(8), stream=stream0)
        buf362 = reinterpret_tensor(buf353, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf353  # reuse
        # Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf355, buf359, buf360, primals_85, primals_86, buf362, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_86
        buf363 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_14], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf362, buf363, 602112, grid=grid(602112), stream=stream0)
        buf364 = buf342; del buf342  # reuse
        buf365 = buf341; del buf341  # reuse
        buf366 = buf340; del buf340  # reuse
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf355, buf363, buf362, primals_87, buf364, buf365, buf366, 4720, 128, grid=grid(4720), stream=stream0)
        buf367 = buf358; del buf358  # reuse
        buf368 = buf357; del buf357  # reuse
        buf369 = buf356; del buf356  # reuse
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf364, buf365, buf366, buf367, buf368, buf369, 80, 59, grid=grid(80), stream=stream0)
        buf370 = buf360; del buf360  # reuse
        buf371 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf938 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_29], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf367, buf368, buf369, buf370, buf371, buf938, 8, 10, grid=grid(8), stream=stream0)
        buf373 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf355, buf363, buf362, primals_87, buf370, buf371, primals_88, primals_89, buf373, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_89
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf375 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf374, primals_282, buf375, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_282
        buf376 = reinterpret_tensor(buf374, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf374  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf375, buf376, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf378 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf379 = buf355; del buf355  # reuse
        # Source Nodes: [mul_28, mul_29, sub_14, x_124, x_129, x_131], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf379, buf377, primals_284, buf363, buf362, primals_87, primals_90, buf378, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_284
        buf380 = buf369; del buf369  # reuse
        buf381 = buf368; del buf368  # reuse
        buf382 = buf367; del buf367  # reuse
        # Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf379, buf380, buf381, buf382, 80, 7527, grid=grid(80), stream=stream0)
        buf383 = buf371; del buf371  # reuse
        buf384 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf937 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_30], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf380, buf381, buf382, buf383, buf384, buf937, 8, 10, grid=grid(8), stream=stream0)
        buf386 = reinterpret_tensor(buf377, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf377  # reuse
        # Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf379, buf383, buf384, primals_91, primals_92, buf386, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_92
        buf387 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_15], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf386, buf387, 602112, grid=grid(602112), stream=stream0)
        buf388 = buf366; del buf366  # reuse
        buf389 = buf365; del buf365  # reuse
        buf390 = buf364; del buf364  # reuse
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf379, buf387, buf386, primals_93, buf388, buf389, buf390, 4720, 128, grid=grid(4720), stream=stream0)
        buf391 = buf382; del buf382  # reuse
        buf392 = buf381; del buf381  # reuse
        buf393 = buf380; del buf380  # reuse
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf388, buf389, buf390, buf391, buf392, buf393, 80, 59, grid=grid(80), stream=stream0)
        buf394 = buf384; del buf384  # reuse
        buf395 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf936 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_31], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf391, buf392, buf393, buf394, buf395, buf936, 8, 10, grid=grid(8), stream=stream0)
        buf397 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf379, buf387, buf386, primals_93, buf394, buf395, primals_94, primals_95, buf397, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_95
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf399 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf398, primals_286, buf399, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_286
        buf400 = reinterpret_tensor(buf398, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf398  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf399, buf400, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf402 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf403 = buf379; del buf379  # reuse
        # Source Nodes: [mul_30, mul_31, sub_15, x_132, x_137, x_139], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf403, buf401, primals_288, buf387, buf386, primals_93, primals_96, buf402, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_288
        buf404 = buf393; del buf393  # reuse
        buf405 = buf392; del buf392  # reuse
        buf406 = buf391; del buf391  # reuse
        # Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf403, buf404, buf405, buf406, 80, 7527, grid=grid(80), stream=stream0)
        buf407 = buf395; del buf395  # reuse
        buf408 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf935 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_32], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf404, buf405, buf406, buf407, buf408, buf935, 8, 10, grid=grid(8), stream=stream0)
        buf410 = reinterpret_tensor(buf401, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf401  # reuse
        # Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf403, buf407, buf408, primals_97, primals_98, buf410, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_98
        buf411 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_16], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf410, buf411, 602112, grid=grid(602112), stream=stream0)
        buf412 = buf390; del buf390  # reuse
        buf413 = buf389; del buf389  # reuse
        buf414 = buf388; del buf388  # reuse
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf403, buf411, buf410, primals_99, buf412, buf413, buf414, 4720, 128, grid=grid(4720), stream=stream0)
        buf415 = buf406; del buf406  # reuse
        buf416 = buf405; del buf405  # reuse
        buf417 = buf404; del buf404  # reuse
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf412, buf413, buf414, buf415, buf416, buf417, 80, 59, grid=grid(80), stream=stream0)
        buf418 = buf408; del buf408  # reuse
        buf419 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf934 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_33], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf415, buf416, buf417, buf418, buf419, buf934, 8, 10, grid=grid(8), stream=stream0)
        buf421 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf403, buf411, buf410, primals_99, buf418, buf419, primals_100, primals_101, buf421, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_101
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf423 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf422, primals_290, buf423, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_290
        buf424 = reinterpret_tensor(buf422, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf422  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf423, buf424, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf426 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf427 = buf403; del buf403  # reuse
        # Source Nodes: [mul_32, mul_33, sub_16, x_140, x_145, x_147], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf427, buf425, primals_292, buf411, buf410, primals_99, primals_102, buf426, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_292
        buf428 = buf417; del buf417  # reuse
        buf429 = buf416; del buf416  # reuse
        buf430 = buf415; del buf415  # reuse
        # Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf427, buf428, buf429, buf430, 80, 7527, grid=grid(80), stream=stream0)
        buf431 = buf419; del buf419  # reuse
        buf432 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf933 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_34], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf428, buf429, buf430, buf431, buf432, buf933, 8, 10, grid=grid(8), stream=stream0)
        buf434 = reinterpret_tensor(buf425, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf425  # reuse
        # Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf427, buf431, buf432, primals_103, primals_104, buf434, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_104
        buf435 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_17], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf434, buf435, 602112, grid=grid(602112), stream=stream0)
        buf436 = buf414; del buf414  # reuse
        buf437 = buf413; del buf413  # reuse
        buf438 = buf412; del buf412  # reuse
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf427, buf435, buf434, primals_105, buf436, buf437, buf438, 4720, 128, grid=grid(4720), stream=stream0)
        buf439 = buf430; del buf430  # reuse
        buf440 = buf429; del buf429  # reuse
        buf441 = buf428; del buf428  # reuse
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf436, buf437, buf438, buf439, buf440, buf441, 80, 59, grid=grid(80), stream=stream0)
        buf442 = buf432; del buf432  # reuse
        buf443 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf932 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_35], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf439, buf440, buf441, buf442, buf443, buf932, 8, 10, grid=grid(8), stream=stream0)
        buf445 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf427, buf435, buf434, primals_105, buf442, buf443, primals_106, primals_107, buf445, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_107
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf447 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf446, primals_294, buf447, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_294
        buf448 = reinterpret_tensor(buf446, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf446  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf447, buf448, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf450 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf451 = buf427; del buf427  # reuse
        # Source Nodes: [mul_34, mul_35, sub_17, x_148, x_153, x_155], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf451, buf449, primals_296, buf435, buf434, primals_105, primals_108, buf450, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_296
        buf452 = buf441; del buf441  # reuse
        buf453 = buf440; del buf440  # reuse
        buf454 = buf439; del buf439  # reuse
        # Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf451, buf452, buf453, buf454, 80, 7527, grid=grid(80), stream=stream0)
        buf455 = buf443; del buf443  # reuse
        buf456 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf931 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_36], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf452, buf453, buf454, buf455, buf456, buf931, 8, 10, grid=grid(8), stream=stream0)
        buf458 = reinterpret_tensor(buf449, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf449  # reuse
        # Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf451, buf455, buf456, primals_109, primals_110, buf458, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_110
        buf459 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_18], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf458, buf459, 602112, grid=grid(602112), stream=stream0)
        buf460 = buf438; del buf438  # reuse
        buf461 = buf437; del buf437  # reuse
        buf462 = buf436; del buf436  # reuse
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf451, buf459, buf458, primals_111, buf460, buf461, buf462, 4720, 128, grid=grid(4720), stream=stream0)
        buf463 = buf454; del buf454  # reuse
        buf464 = buf453; del buf453  # reuse
        buf465 = buf452; del buf452  # reuse
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf460, buf461, buf462, buf463, buf464, buf465, 80, 59, grid=grid(80), stream=stream0)
        buf466 = buf456; del buf456  # reuse
        buf467 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf930 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_37], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf463, buf464, buf465, buf466, buf467, buf930, 8, 10, grid=grid(8), stream=stream0)
        buf469 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf451, buf459, buf458, primals_111, buf466, buf467, primals_112, primals_113, buf469, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_113
        # Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf471 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf470, primals_298, buf471, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_298
        buf472 = reinterpret_tensor(buf470, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf470  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf471, buf472, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf474 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf475 = buf451; del buf451  # reuse
        # Source Nodes: [mul_36, mul_37, sub_18, x_156, x_161, x_163], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf475, buf473, primals_300, buf459, buf458, primals_111, primals_114, buf474, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_300
        buf476 = buf465; del buf465  # reuse
        buf477 = buf464; del buf464  # reuse
        buf478 = buf463; del buf463  # reuse
        # Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf475, buf476, buf477, buf478, 80, 7527, grid=grid(80), stream=stream0)
        buf479 = buf467; del buf467  # reuse
        buf480 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf929 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_38], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf476, buf477, buf478, buf479, buf480, buf929, 8, 10, grid=grid(8), stream=stream0)
        buf482 = reinterpret_tensor(buf473, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf473  # reuse
        # Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf475, buf479, buf480, primals_115, primals_116, buf482, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_116
        buf483 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_19], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf482, buf483, 602112, grid=grid(602112), stream=stream0)
        buf484 = buf462; del buf462  # reuse
        buf485 = buf461; del buf461  # reuse
        buf486 = buf460; del buf460  # reuse
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf475, buf483, buf482, primals_117, buf484, buf485, buf486, 4720, 128, grid=grid(4720), stream=stream0)
        buf487 = buf478; del buf478  # reuse
        buf488 = buf477; del buf477  # reuse
        buf489 = buf476; del buf476  # reuse
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf484, buf485, buf486, buf487, buf488, buf489, 80, 59, grid=grid(80), stream=stream0)
        buf490 = buf480; del buf480  # reuse
        buf491 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf928 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_39], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf487, buf488, buf489, buf490, buf491, buf928, 8, 10, grid=grid(8), stream=stream0)
        buf493 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf475, buf483, buf482, primals_117, buf490, buf491, primals_118, primals_119, buf493, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_119
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(buf493, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf495 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf494, primals_302, buf495, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_302
        buf496 = reinterpret_tensor(buf494, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf494  # reuse
        # Source Nodes: [x_166], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf495, buf496, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf498 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf499 = buf475; del buf475  # reuse
        # Source Nodes: [mul_38, mul_39, sub_19, x_164, x_169, x_171], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf499, buf497, primals_304, buf483, buf482, primals_117, primals_120, buf498, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_304
        buf500 = buf489; del buf489  # reuse
        buf501 = buf488; del buf488  # reuse
        buf502 = buf487; del buf487  # reuse
        # Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf499, buf500, buf501, buf502, 80, 7527, grid=grid(80), stream=stream0)
        buf503 = buf491; del buf491  # reuse
        buf504 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf927 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_40], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf500, buf501, buf502, buf503, buf504, buf927, 8, 10, grid=grid(8), stream=stream0)
        buf506 = reinterpret_tensor(buf497, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf497  # reuse
        # Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf499, buf503, buf504, primals_121, primals_122, buf506, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_122
        buf507 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_20], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf506, buf507, 602112, grid=grid(602112), stream=stream0)
        buf508 = buf486; del buf486  # reuse
        buf509 = buf485; del buf485  # reuse
        buf510 = buf484; del buf484  # reuse
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf499, buf507, buf506, primals_123, buf508, buf509, buf510, 4720, 128, grid=grid(4720), stream=stream0)
        buf511 = buf502; del buf502  # reuse
        buf512 = buf501; del buf501  # reuse
        buf513 = buf500; del buf500  # reuse
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf508, buf509, buf510, buf511, buf512, buf513, 80, 59, grid=grid(80), stream=stream0)
        buf514 = buf504; del buf504  # reuse
        buf515 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf926 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_41], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf511, buf512, buf513, buf514, buf515, buf926, 8, 10, grid=grid(8), stream=stream0)
        buf517 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf499, buf507, buf506, primals_123, buf514, buf515, primals_124, primals_125, buf517, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_125
        # Source Nodes: [x_173], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_305, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf519 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf518, primals_306, buf519, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_306
        buf520 = reinterpret_tensor(buf518, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf518  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf519, buf520, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf520, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf522 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf523 = buf499; del buf499  # reuse
        # Source Nodes: [mul_40, mul_41, sub_20, x_172, x_177, x_179], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf523, buf521, primals_308, buf507, buf506, primals_123, primals_126, buf522, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_308
        buf524 = buf513; del buf513  # reuse
        buf525 = buf512; del buf512  # reuse
        buf526 = buf511; del buf511  # reuse
        # Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf523, buf524, buf525, buf526, 80, 7527, grid=grid(80), stream=stream0)
        buf527 = buf515; del buf515  # reuse
        buf528 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf925 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_42], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf524, buf525, buf526, buf527, buf528, buf925, 8, 10, grid=grid(8), stream=stream0)
        buf530 = reinterpret_tensor(buf521, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf521  # reuse
        # Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf523, buf527, buf528, primals_127, primals_128, buf530, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_128
        buf531 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_21], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf530, buf531, 602112, grid=grid(602112), stream=stream0)
        buf532 = buf510; del buf510  # reuse
        buf533 = buf509; del buf509  # reuse
        buf534 = buf508; del buf508  # reuse
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf523, buf531, buf530, primals_129, buf532, buf533, buf534, 4720, 128, grid=grid(4720), stream=stream0)
        buf535 = buf526; del buf526  # reuse
        buf536 = buf525; del buf525  # reuse
        buf537 = buf524; del buf524  # reuse
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf532, buf533, buf534, buf535, buf536, buf537, 80, 59, grid=grid(80), stream=stream0)
        buf538 = buf528; del buf528  # reuse
        buf539 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf924 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_43], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf535, buf536, buf537, buf538, buf539, buf924, 8, 10, grid=grid(8), stream=stream0)
        buf541 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf523, buf531, buf530, primals_129, buf538, buf539, primals_130, primals_131, buf541, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_131
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf543 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf542, primals_310, buf543, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_310
        buf544 = reinterpret_tensor(buf542, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf542  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf543, buf544, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf546 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf547 = buf523; del buf523  # reuse
        # Source Nodes: [mul_42, mul_43, sub_21, x_180, x_185, x_187], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf547, buf545, primals_312, buf531, buf530, primals_129, primals_132, buf546, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_312
        buf548 = buf537; del buf537  # reuse
        buf549 = buf536; del buf536  # reuse
        buf550 = buf535; del buf535  # reuse
        # Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf547, buf548, buf549, buf550, 80, 7527, grid=grid(80), stream=stream0)
        buf551 = buf539; del buf539  # reuse
        buf552 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf923 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_44], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf548, buf549, buf550, buf551, buf552, buf923, 8, 10, grid=grid(8), stream=stream0)
        buf554 = reinterpret_tensor(buf545, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf545  # reuse
        # Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf547, buf551, buf552, primals_133, primals_134, buf554, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_134
        buf555 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_22], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf554, buf555, 602112, grid=grid(602112), stream=stream0)
        buf556 = buf534; del buf534  # reuse
        buf557 = buf533; del buf533  # reuse
        buf558 = buf532; del buf532  # reuse
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf547, buf555, buf554, primals_135, buf556, buf557, buf558, 4720, 128, grid=grid(4720), stream=stream0)
        buf559 = buf550; del buf550  # reuse
        buf560 = buf549; del buf549  # reuse
        buf561 = buf548; del buf548  # reuse
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf556, buf557, buf558, buf559, buf560, buf561, 80, 59, grid=grid(80), stream=stream0)
        buf562 = buf552; del buf552  # reuse
        buf563 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf922 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_45], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf559, buf560, buf561, buf562, buf563, buf922, 8, 10, grid=grid(8), stream=stream0)
        buf565 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf547, buf555, buf554, primals_135, buf562, buf563, primals_136, primals_137, buf565, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_137
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf567 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf566, primals_314, buf567, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_314
        buf568 = reinterpret_tensor(buf566, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf566  # reuse
        # Source Nodes: [x_190], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf567, buf568, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf568, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf570 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf571 = buf547; del buf547  # reuse
        # Source Nodes: [mul_44, mul_45, sub_22, x_188, x_193, x_195], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf571, buf569, primals_316, buf555, buf554, primals_135, primals_138, buf570, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_316
        buf572 = buf561; del buf561  # reuse
        buf573 = buf560; del buf560  # reuse
        buf574 = buf559; del buf559  # reuse
        # Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf571, buf572, buf573, buf574, 80, 7527, grid=grid(80), stream=stream0)
        buf575 = buf563; del buf563  # reuse
        buf576 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf921 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_46], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf572, buf573, buf574, buf575, buf576, buf921, 8, 10, grid=grid(8), stream=stream0)
        buf578 = reinterpret_tensor(buf569, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf569  # reuse
        # Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf571, buf575, buf576, primals_139, primals_140, buf578, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_140
        buf579 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_23], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf578, buf579, 602112, grid=grid(602112), stream=stream0)
        buf580 = buf558; del buf558  # reuse
        buf581 = buf557; del buf557  # reuse
        buf582 = buf556; del buf556  # reuse
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf571, buf579, buf578, primals_141, buf580, buf581, buf582, 4720, 128, grid=grid(4720), stream=stream0)
        buf583 = buf574; del buf574  # reuse
        buf584 = buf573; del buf573  # reuse
        buf585 = buf572; del buf572  # reuse
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf580, buf581, buf582, buf583, buf584, buf585, 80, 59, grid=grid(80), stream=stream0)
        buf586 = buf576; del buf576  # reuse
        buf587 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf920 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_47], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf583, buf584, buf585, buf586, buf587, buf920, 8, 10, grid=grid(8), stream=stream0)
        buf589 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf571, buf579, buf578, primals_141, buf586, buf587, primals_142, primals_143, buf589, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_143
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(buf589, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf591 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf590, primals_318, buf591, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_318
        buf592 = reinterpret_tensor(buf590, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf590  # reuse
        # Source Nodes: [x_198], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf591, buf592, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_319, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf594 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf595 = buf571; del buf571  # reuse
        # Source Nodes: [mul_46, mul_47, sub_23, x_196, x_201, x_203], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf595, buf593, primals_320, buf579, buf578, primals_141, primals_144, buf594, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_320
        buf596 = buf585; del buf585  # reuse
        buf597 = buf584; del buf584  # reuse
        buf598 = buf583; del buf583  # reuse
        # Source Nodes: [group_norm_48], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf595, buf596, buf597, buf598, 80, 7527, grid=grid(80), stream=stream0)
        buf599 = buf587; del buf587  # reuse
        buf600 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf919 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_48], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf596, buf597, buf598, buf599, buf600, buf919, 8, 10, grid=grid(8), stream=stream0)
        buf602 = reinterpret_tensor(buf593, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf593  # reuse
        # Source Nodes: [group_norm_48], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf595, buf599, buf600, primals_145, primals_146, buf602, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_146
        buf603 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_24], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf602, buf603, 602112, grid=grid(602112), stream=stream0)
        buf604 = buf582; del buf582  # reuse
        buf605 = buf581; del buf581  # reuse
        buf606 = buf580; del buf580  # reuse
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf595, buf603, buf602, primals_147, buf604, buf605, buf606, 4720, 128, grid=grid(4720), stream=stream0)
        buf607 = buf598; del buf598  # reuse
        buf608 = buf597; del buf597  # reuse
        buf609 = buf596; del buf596  # reuse
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf604, buf605, buf606, buf607, buf608, buf609, 80, 59, grid=grid(80), stream=stream0)
        buf610 = buf600; del buf600  # reuse
        buf611 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf918 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_49], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf607, buf608, buf609, buf610, buf611, buf918, 8, 10, grid=grid(8), stream=stream0)
        buf613 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf595, buf603, buf602, primals_147, buf610, buf611, primals_148, primals_149, buf613, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_149
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf615 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf614, primals_322, buf615, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_322
        buf616 = reinterpret_tensor(buf614, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf614  # reuse
        # Source Nodes: [x_206], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf615, buf616, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_323, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf618 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf619 = buf595; del buf595  # reuse
        # Source Nodes: [mul_48, mul_49, sub_24, x_204, x_209, x_211], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf619, buf617, primals_324, buf603, buf602, primals_147, primals_150, buf618, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_324
        buf620 = buf609; del buf609  # reuse
        buf621 = buf608; del buf608  # reuse
        buf622 = buf607; del buf607  # reuse
        # Source Nodes: [group_norm_50], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf619, buf620, buf621, buf622, 80, 7527, grid=grid(80), stream=stream0)
        buf623 = buf611; del buf611  # reuse
        buf624 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf917 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_50], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf620, buf621, buf622, buf623, buf624, buf917, 8, 10, grid=grid(8), stream=stream0)
        buf626 = reinterpret_tensor(buf617, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf617  # reuse
        # Source Nodes: [group_norm_50], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf619, buf623, buf624, primals_151, primals_152, buf626, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_152
        buf627 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_25], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf626, buf627, 602112, grid=grid(602112), stream=stream0)
        buf628 = buf606; del buf606  # reuse
        buf629 = buf605; del buf605  # reuse
        buf630 = buf604; del buf604  # reuse
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf619, buf627, buf626, primals_153, buf628, buf629, buf630, 4720, 128, grid=grid(4720), stream=stream0)
        buf631 = buf622; del buf622  # reuse
        buf632 = buf621; del buf621  # reuse
        buf633 = buf620; del buf620  # reuse
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf628, buf629, buf630, buf631, buf632, buf633, 80, 59, grid=grid(80), stream=stream0)
        buf634 = buf624; del buf624  # reuse
        buf635 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf916 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_51], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf631, buf632, buf633, buf634, buf635, buf916, 8, 10, grid=grid(8), stream=stream0)
        buf637 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf619, buf627, buf626, primals_153, buf634, buf635, primals_154, primals_155, buf637, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_155
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, primals_325, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf639 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf638, primals_326, buf639, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_326
        buf640 = reinterpret_tensor(buf638, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf638  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf639, buf640, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf642 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf643 = buf619; del buf619  # reuse
        # Source Nodes: [mul_50, mul_51, sub_25, x_212, x_217, x_219], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf643, buf641, primals_328, buf627, buf626, primals_153, primals_156, buf642, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_328
        buf644 = buf633; del buf633  # reuse
        buf645 = buf632; del buf632  # reuse
        buf646 = buf631; del buf631  # reuse
        # Source Nodes: [group_norm_52], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf643, buf644, buf645, buf646, 80, 7527, grid=grid(80), stream=stream0)
        buf647 = buf635; del buf635  # reuse
        buf648 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf915 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_52], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf644, buf645, buf646, buf647, buf648, buf915, 8, 10, grid=grid(8), stream=stream0)
        buf650 = reinterpret_tensor(buf641, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf641  # reuse
        # Source Nodes: [group_norm_52], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf643, buf647, buf648, primals_157, primals_158, buf650, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_158
        buf651 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_26], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf650, buf651, 602112, grid=grid(602112), stream=stream0)
        buf652 = buf630; del buf630  # reuse
        buf653 = buf629; del buf629  # reuse
        buf654 = buf628; del buf628  # reuse
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf643, buf651, buf650, primals_159, buf652, buf653, buf654, 4720, 128, grid=grid(4720), stream=stream0)
        buf655 = buf646; del buf646  # reuse
        buf656 = buf645; del buf645  # reuse
        buf657 = buf644; del buf644  # reuse
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf652, buf653, buf654, buf655, buf656, buf657, 80, 59, grid=grid(80), stream=stream0)
        buf658 = buf648; del buf648  # reuse
        buf659 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf914 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_53], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf655, buf656, buf657, buf658, buf659, buf914, 8, 10, grid=grid(8), stream=stream0)
        buf661 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf643, buf651, buf650, primals_159, buf658, buf659, primals_160, primals_161, buf661, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_161
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_329, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf663 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf662, primals_330, buf663, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_330
        buf664 = reinterpret_tensor(buf662, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf662  # reuse
        # Source Nodes: [x_222], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf663, buf664, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf665 = extern_kernels.convolution(buf664, primals_331, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf665, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf666 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf667 = buf643; del buf643  # reuse
        # Source Nodes: [mul_52, mul_53, sub_26, x_220, x_225, x_227], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf667, buf665, primals_332, buf651, buf650, primals_159, primals_162, buf666, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_332
        buf668 = buf657; del buf657  # reuse
        buf669 = buf656; del buf656  # reuse
        buf670 = buf655; del buf655  # reuse
        # Source Nodes: [group_norm_54], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf667, buf668, buf669, buf670, 80, 7527, grid=grid(80), stream=stream0)
        buf671 = buf659; del buf659  # reuse
        buf672 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf913 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_54], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf668, buf669, buf670, buf671, buf672, buf913, 8, 10, grid=grid(8), stream=stream0)
        buf674 = reinterpret_tensor(buf665, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf665  # reuse
        # Source Nodes: [group_norm_54], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf667, buf671, buf672, primals_163, primals_164, buf674, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_164
        buf675 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_27], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf674, buf675, 602112, grid=grid(602112), stream=stream0)
        buf676 = buf654; del buf654  # reuse
        buf677 = buf653; del buf653  # reuse
        buf678 = buf652; del buf652  # reuse
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf667, buf675, buf674, primals_165, buf676, buf677, buf678, 4720, 128, grid=grid(4720), stream=stream0)
        buf679 = buf670; del buf670  # reuse
        buf680 = buf669; del buf669  # reuse
        buf681 = buf668; del buf668  # reuse
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf676, buf677, buf678, buf679, buf680, buf681, 80, 59, grid=grid(80), stream=stream0)
        buf682 = buf672; del buf672  # reuse
        buf683 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf912 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_55], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf679, buf680, buf681, buf682, buf683, buf912, 8, 10, grid=grid(8), stream=stream0)
        buf685 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf667, buf675, buf674, primals_165, buf682, buf683, primals_166, primals_167, buf685, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_167
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf687 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf686, primals_334, buf687, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_334
        buf688 = reinterpret_tensor(buf686, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf686  # reuse
        # Source Nodes: [x_230], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf687, buf688, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf689 = extern_kernels.convolution(buf688, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf689, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf690 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf691 = buf667; del buf667  # reuse
        # Source Nodes: [mul_54, mul_55, sub_27, x_228, x_233, x_235], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf691, buf689, primals_336, buf675, buf674, primals_165, primals_168, buf690, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_336
        buf692 = buf681; del buf681  # reuse
        buf693 = buf680; del buf680  # reuse
        buf694 = buf679; del buf679  # reuse
        # Source Nodes: [group_norm_56], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf691, buf692, buf693, buf694, 80, 7527, grid=grid(80), stream=stream0)
        buf695 = buf683; del buf683  # reuse
        buf696 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf911 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_56], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf692, buf693, buf694, buf695, buf696, buf911, 8, 10, grid=grid(8), stream=stream0)
        buf698 = reinterpret_tensor(buf689, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf689  # reuse
        # Source Nodes: [group_norm_56], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf691, buf695, buf696, primals_169, primals_170, buf698, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_170
        buf699 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_28], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf698, buf699, 602112, grid=grid(602112), stream=stream0)
        buf700 = buf678; del buf678  # reuse
        buf701 = buf677; del buf677  # reuse
        buf702 = buf676; del buf676  # reuse
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf691, buf699, buf698, primals_171, buf700, buf701, buf702, 4720, 128, grid=grid(4720), stream=stream0)
        buf703 = buf694; del buf694  # reuse
        buf704 = buf693; del buf693  # reuse
        buf705 = buf692; del buf692  # reuse
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf700, buf701, buf702, buf703, buf704, buf705, 80, 59, grid=grid(80), stream=stream0)
        buf706 = buf696; del buf696  # reuse
        buf707 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf910 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_57], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf703, buf704, buf705, buf706, buf707, buf910, 8, 10, grid=grid(8), stream=stream0)
        buf709 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf691, buf699, buf698, primals_171, buf706, buf707, primals_172, primals_173, buf709, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_173
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf710 = extern_kernels.convolution(buf709, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf710, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf711 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf710, primals_338, buf711, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_338
        buf712 = reinterpret_tensor(buf710, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf710  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf711, buf712, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf713 = extern_kernels.convolution(buf712, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf713, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf714 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf715 = buf691; del buf691  # reuse
        # Source Nodes: [mul_56, mul_57, sub_28, x_236, x_241, x_243], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_54.run(buf715, buf713, primals_340, buf699, buf698, primals_171, primals_174, buf714, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_340
        buf716 = buf705; del buf705  # reuse
        buf717 = buf704; del buf704  # reuse
        buf718 = buf703; del buf703  # reuse
        # Source Nodes: [group_norm_58], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_50.run(buf715, buf716, buf717, buf718, 80, 7527, grid=grid(80), stream=stream0)
        buf719 = buf707; del buf707  # reuse
        buf720 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf909 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_58], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf716, buf717, buf718, buf719, buf720, buf909, 8, 10, grid=grid(8), stream=stream0)
        buf722 = reinterpret_tensor(buf713, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf713  # reuse
        # Source Nodes: [group_norm_58], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_51.run(buf715, buf719, buf720, primals_175, primals_176, buf722, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_176
        buf723 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_29], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf722, buf723, 602112, grid=grid(602112), stream=stream0)
        buf724 = buf702; del buf702  # reuse
        buf725 = buf701; del buf701  # reuse
        buf726 = buf700; del buf700  # reuse
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_52.run(buf715, buf723, buf722, primals_177, buf724, buf725, buf726, 4720, 128, grid=grid(4720), stream=stream0)
        buf727 = buf718; del buf718  # reuse
        buf728 = buf717; del buf717  # reuse
        buf729 = buf716; del buf716  # reuse
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_41.run(buf724, buf725, buf726, buf727, buf728, buf729, 80, 59, grid=grid(80), stream=stream0)
        del buf724
        del buf725
        del buf726
        buf730 = buf720; del buf720  # reuse
        buf731 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf908 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_59], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_42.run(buf727, buf728, buf729, buf730, buf731, buf908, 8, 10, grid=grid(8), stream=stream0)
        del buf727
        del buf728
        del buf729
        buf733 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_53.run(buf715, buf723, buf722, primals_177, buf730, buf731, primals_178, primals_179, buf733, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_179
        # Source Nodes: [x_245], Original ATen: [aten.convolution]
        buf734 = extern_kernels.convolution(buf733, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf734, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf735 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf734, primals_342, buf735, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del primals_342
        buf736 = reinterpret_tensor(buf734, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf734  # reuse
        # Source Nodes: [x_246], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf735, buf736, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf737 = extern_kernels.convolution(buf736, primals_343, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf737, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf738 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf737, primals_344, buf738, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_344
        buf739 = reinterpret_tensor(buf737, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf737  # reuse
        # Source Nodes: [mul_58, mul_59, sub_29, x_244, x_252], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_55.run(buf715, buf723, buf722, primals_177, buf738, primals_180, buf739, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf715
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf740 = extern_kernels.convolution(buf739, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf740, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf741 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf740, primals_346, buf741, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_346
        buf742 = empty_strided((8, 1, 1, 1, 5, 59), (295, 2360, 2360, 2360, 59, 1), device='cuda', dtype=torch.float32)
        buf743 = empty_strided((8, 1, 1, 1, 5, 59), (295, 2360, 2360, 2360, 59, 1), device='cuda', dtype=torch.float32)
        buf744 = empty_strided((8, 1, 1, 1, 5, 59), (295, 2360, 2360, 2360, 59, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_57.run(buf741, buf742, buf743, buf744, 2360, 128, grid=grid(2360), stream=stream0)
        buf745 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        buf746 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        buf747 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf742, buf743, buf744, buf745, buf746, buf747, 40, 59, grid=grid(40), stream=stream0)
        buf748 = buf731; del buf731  # reuse
        buf749 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf907 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_60], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf745, buf746, buf747, buf748, buf749, buf907, 8, 5, grid=grid(8), stream=stream0)
        buf751 = reinterpret_tensor(buf740, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf740  # reuse
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_60.run(buf741, buf748, buf749, primals_181, primals_182, buf751, 301056, grid=grid(301056), stream=stream0)
        del primals_182
        buf752 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_30], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf751, buf752, 301056, grid=grid(301056), stream=stream0)
        buf753 = buf744; del buf744  # reuse
        buf754 = buf743; del buf743  # reuse
        buf755 = buf742; del buf742  # reuse
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_62.run(buf741, buf752, buf751, primals_183, buf753, buf754, buf755, 2360, 128, grid=grid(2360), stream=stream0)
        buf756 = buf747; del buf747  # reuse
        buf757 = buf746; del buf746  # reuse
        buf758 = buf745; del buf745  # reuse
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf753, buf754, buf755, buf756, buf757, buf758, 40, 59, grid=grid(40), stream=stream0)
        buf759 = buf749; del buf749  # reuse
        buf760 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf906 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_61], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf756, buf757, buf758, buf759, buf760, buf906, 8, 5, grid=grid(8), stream=stream0)
        buf762 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_63.run(buf741, buf752, buf751, primals_183, buf759, buf760, primals_184, primals_185, buf762, 301056, grid=grid(301056), stream=stream0)
        del primals_185
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf763 = extern_kernels.convolution(buf762, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf763, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf764 = reinterpret_tensor(buf278, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf278  # reuse
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf763, primals_348, buf764, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_348
        buf765 = reinterpret_tensor(buf763, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf763  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf764, buf765, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf765, primals_349, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf767 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf766, primals_350, buf767, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_350
        buf768 = buf766; del buf766  # reuse
        # Source Nodes: [mul_60, mul_61, sub_30, x_256, x_263], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_66.run(buf741, buf752, buf751, primals_183, buf767, primals_186, buf768, 392, 768, grid=grid(392, 768), stream=stream0)
        buf769 = buf758; del buf758  # reuse
        buf770 = buf757; del buf757  # reuse
        buf771 = buf756; del buf756  # reuse
        # Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_67.run(buf768, buf769, buf770, buf771, 40, 7527, grid=grid(40), stream=stream0)
        buf772 = buf760; del buf760  # reuse
        buf773 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf905 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_62], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf769, buf770, buf771, buf772, buf773, buf905, 8, 5, grid=grid(8), stream=stream0)
        buf775 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_68.run(buf768, buf772, buf773, primals_187, primals_188, buf775, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_188
        buf776 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_31], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf775, buf776, 301056, grid=grid(301056), stream=stream0)
        buf777 = buf755; del buf755  # reuse
        buf778 = buf754; del buf754  # reuse
        buf779 = buf753; del buf753  # reuse
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_69.run(buf768, buf776, buf775, primals_189, buf777, buf778, buf779, 2360, 128, grid=grid(2360), stream=stream0)
        buf780 = buf771; del buf771  # reuse
        buf781 = buf770; del buf770  # reuse
        buf782 = buf769; del buf769  # reuse
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf777, buf778, buf779, buf780, buf781, buf782, 40, 59, grid=grid(40), stream=stream0)
        buf783 = buf773; del buf773  # reuse
        buf784 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf904 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_63], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf780, buf781, buf782, buf783, buf784, buf904, 8, 5, grid=grid(8), stream=stream0)
        buf786 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_70.run(buf768, buf776, buf775, primals_189, buf783, buf784, primals_190, primals_191, buf786, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_191
        # Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf787 = extern_kernels.convolution(buf786, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf787, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf788 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf787, primals_352, buf788, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_352
        buf789 = reinterpret_tensor(buf787, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf787  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf788, buf789, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf790 = extern_kernels.convolution(buf789, primals_353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf790, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf791 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf792 = buf768; del buf768  # reuse
        # Source Nodes: [mul_62, mul_63, sub_31, x_264, x_269, x_271], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_71.run(buf792, buf790, primals_354, buf776, buf775, primals_189, primals_192, buf791, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_354
        buf793 = buf782; del buf782  # reuse
        buf794 = buf781; del buf781  # reuse
        buf795 = buf780; del buf780  # reuse
        # Source Nodes: [group_norm_64], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_67.run(buf792, buf793, buf794, buf795, 40, 7527, grid=grid(40), stream=stream0)
        buf796 = buf784; del buf784  # reuse
        buf797 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf903 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_64], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf793, buf794, buf795, buf796, buf797, buf903, 8, 5, grid=grid(8), stream=stream0)
        buf799 = reinterpret_tensor(buf790, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf790  # reuse
        # Source Nodes: [group_norm_64], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_68.run(buf792, buf796, buf797, primals_193, primals_194, buf799, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_194
        buf800 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_32], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf799, buf800, 301056, grid=grid(301056), stream=stream0)
        buf801 = buf779; del buf779  # reuse
        buf802 = buf778; del buf778  # reuse
        buf803 = buf777; del buf777  # reuse
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_69.run(buf792, buf800, buf799, primals_195, buf801, buf802, buf803, 2360, 128, grid=grid(2360), stream=stream0)
        buf804 = buf795; del buf795  # reuse
        buf805 = buf794; del buf794  # reuse
        buf806 = buf793; del buf793  # reuse
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf801, buf802, buf803, buf804, buf805, buf806, 40, 59, grid=grid(40), stream=stream0)
        buf807 = buf797; del buf797  # reuse
        buf808 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf902 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_65], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf804, buf805, buf806, buf807, buf808, buf902, 8, 5, grid=grid(8), stream=stream0)
        buf810 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_70.run(buf792, buf800, buf799, primals_195, buf807, buf808, primals_196, primals_197, buf810, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_197
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf811 = extern_kernels.convolution(buf810, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf811, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf812 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf811, primals_356, buf812, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_356
        buf813 = reinterpret_tensor(buf811, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf811  # reuse
        # Source Nodes: [x_274], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf812, buf813, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf814 = extern_kernels.convolution(buf813, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf814, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf815 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf816 = buf792; del buf792  # reuse
        # Source Nodes: [mul_64, mul_65, sub_32, x_272, x_277, x_279], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_71.run(buf816, buf814, primals_358, buf800, buf799, primals_195, primals_198, buf815, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_358
        buf817 = buf806; del buf806  # reuse
        buf818 = buf805; del buf805  # reuse
        buf819 = buf804; del buf804  # reuse
        # Source Nodes: [group_norm_66], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_67.run(buf816, buf817, buf818, buf819, 40, 7527, grid=grid(40), stream=stream0)
        buf820 = buf808; del buf808  # reuse
        buf821 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf901 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_66], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf817, buf818, buf819, buf820, buf821, buf901, 8, 5, grid=grid(8), stream=stream0)
        buf823 = reinterpret_tensor(buf814, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf814  # reuse
        # Source Nodes: [group_norm_66], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_68.run(buf816, buf820, buf821, primals_199, primals_200, buf823, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_200
        buf824 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_33], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf823, buf824, 301056, grid=grid(301056), stream=stream0)
        buf825 = buf803; del buf803  # reuse
        buf826 = buf802; del buf802  # reuse
        buf827 = buf801; del buf801  # reuse
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_69.run(buf816, buf824, buf823, primals_201, buf825, buf826, buf827, 2360, 128, grid=grid(2360), stream=stream0)
        buf828 = buf819; del buf819  # reuse
        buf829 = buf818; del buf818  # reuse
        buf830 = buf817; del buf817  # reuse
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf825, buf826, buf827, buf828, buf829, buf830, 40, 59, grid=grid(40), stream=stream0)
        buf831 = buf821; del buf821  # reuse
        buf832 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf900 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_67], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf828, buf829, buf830, buf831, buf832, buf900, 8, 5, grid=grid(8), stream=stream0)
        buf834 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_70.run(buf816, buf824, buf823, primals_201, buf831, buf832, primals_202, primals_203, buf834, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_203
        # Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf834, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf836 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf835, primals_360, buf836, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_360
        buf837 = reinterpret_tensor(buf835, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf835  # reuse
        # Source Nodes: [x_282], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf836, buf837, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf838 = extern_kernels.convolution(buf837, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf838, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf839 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf840 = buf816; del buf816  # reuse
        # Source Nodes: [mul_66, mul_67, sub_33, x_280, x_285, x_287], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_71.run(buf840, buf838, primals_362, buf824, buf823, primals_201, primals_204, buf839, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_362
        buf841 = buf830; del buf830  # reuse
        buf842 = buf829; del buf829  # reuse
        buf843 = buf828; del buf828  # reuse
        # Source Nodes: [group_norm_68], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_67.run(buf840, buf841, buf842, buf843, 40, 7527, grid=grid(40), stream=stream0)
        buf844 = buf832; del buf832  # reuse
        buf845 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf899 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_68], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf841, buf842, buf843, buf844, buf845, buf899, 8, 5, grid=grid(8), stream=stream0)
        buf847 = reinterpret_tensor(buf838, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf838  # reuse
        # Source Nodes: [group_norm_68], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_68.run(buf840, buf844, buf845, primals_205, primals_206, buf847, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_206
        buf848 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_34], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf847, buf848, 301056, grid=grid(301056), stream=stream0)
        buf849 = buf827; del buf827  # reuse
        buf850 = buf826; del buf826  # reuse
        buf851 = buf825; del buf825  # reuse
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_69.run(buf840, buf848, buf847, primals_207, buf849, buf850, buf851, 2360, 128, grid=grid(2360), stream=stream0)
        buf852 = buf843; del buf843  # reuse
        buf853 = buf842; del buf842  # reuse
        buf854 = buf841; del buf841  # reuse
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf849, buf850, buf851, buf852, buf853, buf854, 40, 59, grid=grid(40), stream=stream0)
        buf855 = buf845; del buf845  # reuse
        buf856 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf898 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_69], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf852, buf853, buf854, buf855, buf856, buf898, 8, 5, grid=grid(8), stream=stream0)
        buf858 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_70.run(buf840, buf848, buf847, primals_207, buf855, buf856, primals_208, primals_209, buf858, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_209
        # Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf859 = extern_kernels.convolution(buf858, primals_363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf859, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf860 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf859, primals_364, buf860, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_364
        buf861 = reinterpret_tensor(buf859, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf859  # reuse
        # Source Nodes: [x_290], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf860, buf861, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf862 = extern_kernels.convolution(buf861, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf862, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf863 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf864 = buf840; del buf840  # reuse
        # Source Nodes: [mul_68, mul_69, sub_34, x_288, x_293, x_295], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_convolution_mul_sub_71.run(buf864, buf862, primals_366, buf848, buf847, primals_207, primals_210, buf863, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_366
        buf865 = buf854; del buf854  # reuse
        buf866 = buf853; del buf853  # reuse
        buf867 = buf852; del buf852  # reuse
        # Source Nodes: [group_norm_70], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_67.run(buf864, buf865, buf866, buf867, 40, 7527, grid=grid(40), stream=stream0)
        buf868 = buf856; del buf856  # reuse
        buf869 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf897 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_70], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf865, buf866, buf867, buf868, buf869, buf897, 8, 5, grid=grid(8), stream=stream0)
        buf871 = reinterpret_tensor(buf862, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf862  # reuse
        # Source Nodes: [group_norm_70], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_68.run(buf864, buf868, buf869, primals_211, primals_212, buf871, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_212
        buf872 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [y_35], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_61.run(buf871, buf872, 301056, grid=grid(301056), stream=stream0)
        buf873 = buf851; del buf851  # reuse
        buf874 = buf850; del buf850  # reuse
        buf875 = buf849; del buf849  # reuse
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_69.run(buf864, buf872, buf871, primals_213, buf873, buf874, buf875, 2360, 128, grid=grid(2360), stream=stream0)
        buf876 = buf867; del buf867  # reuse
        buf877 = buf866; del buf866  # reuse
        buf878 = buf865; del buf865  # reuse
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_58.run(buf873, buf874, buf875, buf876, buf877, buf878, 40, 59, grid=grid(40), stream=stream0)
        del buf873
        del buf874
        del buf875
        buf879 = buf869; del buf869  # reuse
        buf880 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf896 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_71], Original ATen: [aten.detach, aten.native_group_norm]
        triton_per_fused_detach_native_group_norm_59.run(buf876, buf877, buf878, buf879, buf880, buf896, 8, 5, grid=grid(8), stream=stream0)
        del buf876
        del buf877
        del buf878
        buf882 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_70.run(buf864, buf872, buf871, primals_213, buf879, buf880, primals_214, primals_215, buf882, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_215
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf884 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf883, primals_368, buf884, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del primals_368
        buf885 = reinterpret_tensor(buf883, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf883  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf884, buf885, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf886 = extern_kernels.convolution(buf885, primals_369, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf886, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf887 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf888 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_70, mul_71, sub_35, x_296, x_301, x_306, x_307], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.sub]
        triton_per_fused_add_convolution_mean_mul_sub_72.run(buf886, primals_370, buf864, buf872, buf871, primals_213, primals_216, buf887, buf888, 6144, 49, grid=grid(6144), stream=stream0)
        del buf864
        del buf886
        del primals_370
        buf892 = empty_strided((8, 1, 1, 768), (768, 1, 768, 1), device='cuda', dtype=torch.float32)
        buf893 = empty((8, 768), device='cuda', dtype=torch.float32)
        buf895 = reinterpret_tensor(buf880, (8, 1, 1, 1), (1, 1, 1, 1), 0); del buf880  # reuse
        # Source Nodes: [x_311, x_314], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73.run(buf888, primals_217, primals_218, buf892, buf893, buf895, 8, 768, grid=grid(8), stream=stream0)
        del buf888
        del primals_218
        buf894 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_372, buf893, reinterpret_tensor(primals_371, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf894)
        del primals_372
        return (buf894, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, buf0, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, buf1, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, buf2, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, buf3, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, buf4, buf6, buf16, buf17, buf27, buf29, buf30, buf32, buf40, buf41, buf51, buf53, buf54, buf56, buf64, buf65, buf75, buf77, buf78, buf80, buf88, buf89, buf99, buf101, buf102, buf104, buf112, buf113, buf123, buf125, buf126, buf128, buf136, buf137, buf147, buf149, buf150, buf152, buf153, buf155, buf165, buf166, buf176, buf178, buf179, buf181, buf189, buf190, buf200, buf202, buf203, buf205, buf213, buf214, buf224, buf226, buf227, buf229, buf237, buf238, buf248, buf250, buf251, buf253, buf261, buf262, buf272, buf274, buf275, buf277, buf285, buf286, buf296, buf298, buf299, buf301, buf302, buf304, buf314, buf315, buf325, buf327, buf328, buf330, buf338, buf339, buf349, buf351, buf352, buf354, buf362, buf363, buf373, buf375, buf376, buf378, buf386, buf387, buf397, buf399, buf400, buf402, buf410, buf411, buf421, buf423, buf424, buf426, buf434, buf435, buf445, buf447, buf448, buf450, buf458, buf459, buf469, buf471, buf472, buf474, buf482, buf483, buf493, buf495, buf496, buf498, buf506, buf507, buf517, buf519, buf520, buf522, buf530, buf531, buf541, buf543, buf544, buf546, buf554, buf555, buf565, buf567, buf568, buf570, buf578, buf579, buf589, buf591, buf592, buf594, buf602, buf603, buf613, buf615, buf616, buf618, buf626, buf627, buf637, buf639, buf640, buf642, buf650, buf651, buf661, buf663, buf664, buf666, buf674, buf675, buf685, buf687, buf688, buf690, buf698, buf699, buf709, buf711, buf712, buf714, buf722, buf723, buf733, buf735, buf736, buf738, buf739, buf741, buf751, buf752, buf762, buf764, buf765, buf767, buf775, buf776, buf786, buf788, buf789, buf791, buf799, buf800, buf810, buf812, buf813, buf815, buf823, buf824, buf834, buf836, buf837, buf839, buf847, buf848, buf858, buf860, buf861, buf863, buf871, buf872, buf882, buf884, buf885, buf887, buf892, buf893, reinterpret_tensor(primals_371, (1000, 768), (768, 1), 0), buf895, reinterpret_tensor(buf879, (8, 1), (1, 1), 0), buf896, reinterpret_tensor(buf868, (8, 1), (1, 1), 0), buf897, reinterpret_tensor(buf855, (8, 1), (1, 1), 0), buf898, reinterpret_tensor(buf844, (8, 1), (1, 1), 0), buf899, reinterpret_tensor(buf831, (8, 1), (1, 1), 0), buf900, reinterpret_tensor(buf820, (8, 1), (1, 1), 0), buf901, reinterpret_tensor(buf807, (8, 1), (1, 1), 0), buf902, reinterpret_tensor(buf796, (8, 1), (1, 1), 0), buf903, reinterpret_tensor(buf783, (8, 1), (1, 1), 0), buf904, reinterpret_tensor(buf772, (8, 1), (1, 1), 0), buf905, reinterpret_tensor(buf759, (8, 1), (1, 1), 0), buf906, reinterpret_tensor(buf748, (8, 1), (1, 1), 0), buf907, reinterpret_tensor(buf730, (8, 1), (1, 1), 0), buf908, reinterpret_tensor(buf719, (8, 1), (1, 1), 0), buf909, reinterpret_tensor(buf706, (8, 1), (1, 1), 0), buf910, reinterpret_tensor(buf695, (8, 1), (1, 1), 0), buf911, reinterpret_tensor(buf682, (8, 1), (1, 1), 0), buf912, reinterpret_tensor(buf671, (8, 1), (1, 1), 0), buf913, reinterpret_tensor(buf658, (8, 1), (1, 1), 0), buf914, reinterpret_tensor(buf647, (8, 1), (1, 1), 0), buf915, reinterpret_tensor(buf634, (8, 1), (1, 1), 0), buf916, reinterpret_tensor(buf623, (8, 1), (1, 1), 0), buf917, reinterpret_tensor(buf610, (8, 1), (1, 1), 0), buf918, reinterpret_tensor(buf599, (8, 1), (1, 1), 0), buf919, reinterpret_tensor(buf586, (8, 1), (1, 1), 0), buf920, reinterpret_tensor(buf575, (8, 1), (1, 1), 0), buf921, reinterpret_tensor(buf562, (8, 1), (1, 1), 0), buf922, reinterpret_tensor(buf551, (8, 1), (1, 1), 0), buf923, reinterpret_tensor(buf538, (8, 1), (1, 1), 0), buf924, reinterpret_tensor(buf527, (8, 1), (1, 1), 0), buf925, reinterpret_tensor(buf514, (8, 1), (1, 1), 0), buf926, reinterpret_tensor(buf503, (8, 1), (1, 1), 0), buf927, reinterpret_tensor(buf490, (8, 1), (1, 1), 0), buf928, reinterpret_tensor(buf479, (8, 1), (1, 1), 0), buf929, reinterpret_tensor(buf466, (8, 1), (1, 1), 0), buf930, reinterpret_tensor(buf455, (8, 1), (1, 1), 0), buf931, reinterpret_tensor(buf442, (8, 1), (1, 1), 0), buf932, reinterpret_tensor(buf431, (8, 1), (1, 1), 0), buf933, reinterpret_tensor(buf418, (8, 1), (1, 1), 0), buf934, reinterpret_tensor(buf407, (8, 1), (1, 1), 0), buf935, reinterpret_tensor(buf394, (8, 1), (1, 1), 0), buf936, reinterpret_tensor(buf383, (8, 1), (1, 1), 0), buf937, reinterpret_tensor(buf370, (8, 1), (1, 1), 0), buf938, reinterpret_tensor(buf359, (8, 1), (1, 1), 0), buf939, reinterpret_tensor(buf346, (8, 1), (1, 1), 0), buf940, reinterpret_tensor(buf335, (8, 1), (1, 1), 0), buf941, reinterpret_tensor(buf322, (8, 1), (1, 1), 0), buf942, reinterpret_tensor(buf311, (8, 1), (1, 1), 0), buf943, reinterpret_tensor(buf293, (8, 1), (1, 1), 0), buf944, reinterpret_tensor(buf282, (8, 1), (1, 1), 0), buf945, reinterpret_tensor(buf269, (8, 1), (1, 1), 0), buf946, reinterpret_tensor(buf258, (8, 1), (1, 1), 0), buf947, reinterpret_tensor(buf245, (8, 1), (1, 1), 0), buf948, reinterpret_tensor(buf234, (8, 1), (1, 1), 0), buf949, reinterpret_tensor(buf221, (8, 1), (1, 1), 0), buf950, reinterpret_tensor(buf210, (8, 1), (1, 1), 0), buf951, reinterpret_tensor(buf197, (8, 1), (1, 1), 0), buf952, reinterpret_tensor(buf186, (8, 1), (1, 1), 0), buf953, reinterpret_tensor(buf173, (8, 1), (1, 1), 0), buf954, reinterpret_tensor(buf162, (8, 1), (1, 1), 0), buf955, reinterpret_tensor(buf144, (8, 1), (1, 1), 0), buf956, reinterpret_tensor(buf133, (8, 1), (1, 1), 0), buf957, reinterpret_tensor(buf120, (8, 1), (1, 1), 0), buf958, reinterpret_tensor(buf109, (8, 1), (1, 1), 0), buf959, reinterpret_tensor(buf96, (8, 1), (1, 1), 0), buf960, reinterpret_tensor(buf85, (8, 1), (1, 1), 0), buf961, reinterpret_tensor(buf72, (8, 1), (1, 1), 0), buf962, reinterpret_tensor(buf61, (8, 1), (1, 1), 0), buf963, reinterpret_tensor(buf48, (8, 1), (1, 1), 0), buf964, reinterpret_tensor(buf37, (8, 1), (1, 1), 0), buf965, reinterpret_tensor(buf24, (8, 1), (1, 1), 0), buf966, reinterpret_tensor(buf13, (8, 1), (1, 1), 0), buf967, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_15 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_119 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((96, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('poolformer_m36', benchmark_compiled_module)
