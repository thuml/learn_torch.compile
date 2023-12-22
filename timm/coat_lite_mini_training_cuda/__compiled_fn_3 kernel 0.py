
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


# kernel path: /tmp/torchinductor_youkaichao/fo/cfodgusapprr525kundrvbwdgdef2gkdp2e3sxhzw2pqzxdoxeua.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyzoqyfvrlewtn2yfrkwpbssyolc4jv7wqulj3brrjfpjwciezj.py
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
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (256*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57nribfkbakrh7fdc3za6ywogs5dqa7a7owpu62r2ugx2jqip5z.py
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
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhfqc7mkprcf4qbmf7x5tqa6wmhwccrch2ht4n2rldb3yz3ulf5.py
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
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 163840
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4vwvvdx2gbdh7mnlevelhclghf7u4vro5rqdmfv3pce7osp5ey.py
# Source Nodes: [x1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x1 => add, clone, mul, rsqrt, sub, var_mean
triton_per_fused_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp26 = tmp24 / tmp20
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbs7y6jr25mdoakq76hayrijc2yonzx3l4m3yg2iiyo5g56gi3r.py
# Source Nodes: [cat_39], Original ATen: [aten.cat]
# cat_39 => cat
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 3137
    x0 = xindex % 64
    x2 = (xindex // 200768)
    x3 = xindex % 200768
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-64) + x3 + (200704*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr3 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cig6y2v4hlhekmkex6fojf5ztlx76zon7fjzumfgacuuvdw3p7a2.py
# Source Nodes: [cat_38], Original ATen: [aten.cat]
# cat_38 => cat_1
triton_poi_fused_cat_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3137
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
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (64*x2) + (200768*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*y3) + (((-1) + x2) % 3136)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (64 + y0 + (64*(((-1) + x2) % 3136)) + (200768*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (64*x2) + (200768*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7h6brp4h6puf5e5daba5mrusv4iki4hii6tqbaqbf4v4uy5yd7e.py
# Source Nodes: [cur, l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
# cur => add_3, add_4, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# l__mod___serial_blocks1_0_factoratt_crpe_qkv => view_3
triton_per_fused_native_layer_norm_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (64*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvhvzfideydh4s46lou7qneoblf5jkwa2bgp7v4vwd3b2vp4m6v.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => amax, clone_1
triton_red_fused__softmax_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57ffb6ojfck24aca67gffdl5pfntfx57hjoij3fnavdvdk6csfz.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => amax, clone_1
triton_per_fused__softmax_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvn4uzk63333eukiwefktxolfrwfytbvfe4ftcekwjr3op46cbzd.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => clone_1, exp, sub_2, sum_1
triton_red_fused__softmax_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (64*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbpduxsmna2eorxprfs6qvwln7b2qsk7rt5zugcskyh47nhzsbh.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => clone_1, exp, sub_2, sum_1
triton_per_fused__softmax_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjphw2r7ncxc7bl422rdg4h2ik2sm7taa5rgln7g6nia2mub7r4.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax, aten.detach]
# k_softmax => clone_1, div, exp, sub_2
triton_poi_fused__softmax_detach_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_detach_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 25096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 8
    y1 = (yindex // 8)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x2 + (8*y0) + (192*x3) + (602304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (8*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (8*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5 + (25096*y4)), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (8*x5) + (200768*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5xme4osasz3ppuzyivs77yxnz66znf7o3cwycsaff6i3phuy4e.py
# Source Nodes: [factor_att], Original ATen: [aten.clone]
# factor_att => clone_2
triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5ebp5vehw2hhcihirchjaqsf6xacywobzermqt6nfonb4ulu2u.py
# Source Nodes: [factor_att_1], Original ATen: [aten.clone]
# factor_att_1 => clone_3
triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/coubhyw676jjhoz6urx3v4pozcmwjfy2lsq3irfpfkb5g7o2hxq3.py
# Source Nodes: [cat_37], Original ATen: [aten.cat]
# cat_37 => cat_2
triton_poi_fused_cat_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 64
    x2 = xindex
    y1 = (yindex // 64)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (50176*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 40, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + ((-50176) + x2 + (3136*y0) + (75264*y1)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-16) + y0, [XBLOCK, YBLOCK])), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1, 1], 64, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((-125440) + x2 + (3136*y0) + (75264*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-40) + y0, [XBLOCK, YBLOCK])), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp28, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjk442mizfy4xwpbdadis2emi6m5b6w6sm7niuwhuypv5qylh3ql.py
# Source Nodes: [x_11], Original ATen: [aten.view]
# x_11 => view_15
triton_poi_fused_view_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(x1 % 3137)) + (25096*(x0 // 8)) + (200768*(x1 // 3137)) + (x0 % 8)), xmask)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + (x1 % 3137)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (192*x1)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (64*(((-1) + (x1 % 3137)) % 3136)) + (200704*(x1 // 3137))), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xdnqamiy4ve2xzda7w5biz7hjcazlja2at4e7sfmp3hl27j7zv.py
# Source Nodes: [cur_2, x_13, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cur_2 => add_7, add_8, mul_6, mul_7, rsqrt_2, sub_3, var_mean_2
# x_13 => add_6
# x_15 => view_17
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
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
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvppadaewy5kbuswyhqhk43tkdr5gn7nsia4onm3zky3ktbuxik.py
# Source Nodes: [x_16, x_19], Original ATen: [aten.gelu, aten.view]
# x_16 => add_9, erf, mul_10, mul_8, mul_9
# x_19 => view_19
triton_poi_fused_gelu_view_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12849152
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzjjm6dxblm35iyoxgdavsavvb4eyqresxv26cx7ms33cpny7yu.py
# Source Nodes: [x1_2, x_13], Original ATen: [aten.add]
# x1_2 => add_10
# x_13 => add_6
triton_poi_fused_add_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6smb5wjbnb25n3o2nejeajmechbwgkugp5kzbwsb7tew2z7zut.py
# Source Nodes: [cat_36], Original ATen: [aten.cat]
# cat_36 => cat_3
triton_poi_fused_cat_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3137
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
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0 + (200768*y1), [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*y3) + (((-1) + x2) % 3136)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (64 + y0 + (64*(((-1) + x2) % 3136)) + (200768*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (64*x2) + (200768*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqpzg2e5l4he6kd75cxslvulwu6hxxbrydrleusqoxvmqqhsbd3.py
# Source Nodes: [x1_nocls], Original ATen: [aten.clone]
# x1_nocls => clone_15
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 200704)
    x3 = xindex % 200704
    x0 = xindex % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x3 + (200768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (64 + x3 + (200768*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (64 + x3 + (200768*x2)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celvzl4bbqj7bcb3qzclsjosfxr5c3jc4zyhhaylw6pgaekfi2io.py
# Source Nodes: [x2], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x2 => add_20, clone_16, mul_20, rsqrt_5, sub_7, var_mean_5
triton_red_fused_native_layer_norm_native_layer_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, rmask & xmask)
    tmp17 = 128.0
    tmp18 = tmp5 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp21 / tmp17
    tl.store(out_ptr3 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdy63osbwp6csnq66qtyxm2d7zx47rpdcve77q5c4jagkzwbnjpq.py
# Source Nodes: [cat_34], Original ATen: [aten.cat]
# cat_34 => cat_5
triton_poi_fused_cat_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 785
    x0 = xindex % 128
    x2 = (xindex // 100480)
    x3 = xindex % 100480
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-128) + x3 + (100352*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr3 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm33p67trhsqqlhbjwe4i5pzf3jz7fbrx6natorudnl7c5n722au.py
# Source Nodes: [cat_33], Original ATen: [aten.cat]
# cat_33 => cat_6
triton_poi_fused_cat_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 785
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (128*x2) + (100480*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*y3) + (((-1) + x2) % 784)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (128 + y0 + (128*(((-1) + x2) % 784)) + (100480*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (128*x2) + (100480*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52m4dkr4l5crih7wfty5zrcns2exvamho6qfkq3nnirjft77htg.py
# Source Nodes: [cur_8, l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
# cur_8 => add_23, add_24, mul_22, mul_23, rsqrt_6, sub_8, var_mean_6
# l__mod___serial_blocks2_0_factoratt_crpe_qkv => view_45
triton_per_fused_native_layer_norm_view_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2lt3rlbwle4rkkb7ekhlhm2bndnh37qz5uwilk7xbacj6rgitj.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => amax_2, clone_17
triton_red_fused__softmax_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6anbqcgcv2rbcvpd5k6su24ogq6wadqor4etolaah5tfs6eekj.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => amax_2, clone_17
triton_per_fused__softmax_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tetkwfud4pnm3iskjhll3jtc4ijsau54f5oq774aej5ozo3el4.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => clone_17, exp_2, sub_9, sum_3
triton_red_fused__softmax_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (128*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgb2inl3wrpq7zern5e7x53cxulzicbgrxbn3canyk5vc562jcjy.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => clone_17, exp_2, sub_9, sum_3
triton_per_fused__softmax_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tbxw3ale2zspxbbm3shxn46whc6zdiugmk6ncwln5t7hf6yoqj.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax, aten.detach]
# k_softmax_2 => clone_17, div_2, exp_2, sub_9
triton_poi_fused__softmax_detach_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_detach_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12560
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 8
    y1 = (yindex // 8)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x2 + (16*y0) + (384*x3) + (301440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (16*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (16*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5 + (12560*y4)), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (8*x5) + (100480*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrjagn75spnjopwws5spoqlnnfa5p43hscquomtxis3j6hblz6o.py
# Source Nodes: [factor_att_4], Original ATen: [aten.clone]
# factor_att_4 => clone_18
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cysqpb7y3xxinguptsojqkc5ixdfy7sejd74mgg7dy7xmak7oc42.py
# Source Nodes: [factor_att_5], Original ATen: [aten.clone]
# factor_att_5 => clone_19
triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqj3ufcsbylt6aycyyfhrqdxzzmxdpcjoih3pizlstyx5d6pbyn.py
# Source Nodes: [cat_32], Original ATen: [aten.cat]
# cat_32 => cat_7
triton_poi_fused_cat_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
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
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (25088*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 80, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + ((-25088) + x2 + (784*y0) + (37632*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-32) + y0, [XBLOCK, YBLOCK])), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1, 1], 128, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((-62720) + x2 + (784*y0) + (37632*y1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4liy24ulillayo7itkll7ksxv6jiivmhd5ahkbci5nrsxtqqsb.py
# Source Nodes: [x_51], Original ATen: [aten.view]
# x_51 => view_57
triton_poi_fused_view_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*(x1 % 785)) + (12560*(x0 // 16)) + (100480*(x1 // 785)) + (x0 % 16)), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + (x1 % 785)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (384*x1)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (128*(((-1) + (x1 % 785)) % 784)) + (100352*(x1 // 785))), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvx2kspzgcselqjentjirp5m4eqbgkvgizqgvxl4olvous5posj.py
# Source Nodes: [cur_10, x_53, x_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cur_10 => add_27, add_28, mul_26, mul_27, rsqrt_7, sub_10, var_mean_7
# x_53 => add_26
# x_55 => view_59
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rpffm3g7cqwplyz7k3y7kzpaskgndcnaxnbx3pm5egwa7fo7oq.py
# Source Nodes: [x_56, x_59], Original ATen: [aten.gelu, aten.view]
# x_56 => add_29, erf_2, mul_28, mul_29, mul_30
# x_59 => view_61
triton_poi_fused_gelu_view_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6430720
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqyjodyufuqngdjvxz3qssk5pvxluxfty53ed2azjmawwy2vruj.py
# Source Nodes: [x2_2, x_53], Original ATen: [aten.add]
# x2_2 => add_30
# x_53 => add_26
triton_poi_fused_add_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbk4ygqrrdhm4jy5ddpdhatptt2l2iazm2yxdqf4b5dztnjltcx.py
# Source Nodes: [cat_31], Original ATen: [aten.cat]
# cat_31 => cat_8
triton_poi_fused_cat_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 785
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0 + (100480*y1), [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*y3) + (((-1) + x2) % 784)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (128 + y0 + (128*(((-1) + x2) % 784)) + (100480*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (128*x2) + (100480*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdypgjadpuyj22uouz7dmu6ktmnpktld6t6u2s65qf7kdjv5ooi.py
# Source Nodes: [x2_nocls], Original ATen: [aten.clone]
# x2_nocls => clone_31
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 100352)
    x3 = xindex % 100352
    x0 = xindex % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x3 + (100480*x2)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x3 + (100480*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (128 + x3 + (100480*x2)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2bltrp44qfpwytelyns3gfto75v2k47ytioxa5aon2bmzo45rd.py
# Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
# x3 => clone_32, var_mean_10
triton_red_fused_native_layer_norm_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 196) % 3
    x0 = xindex % 196
    x2 = (xindex // 588)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (196*r3) + (20972*x1) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp17, xmask)
    tl.store(out_ptr1 + (x4), tmp18, xmask)
    tl.store(out_ptr2 + (x4), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmo7ubylk3fhbzkjr7jqp7a5l5hp27k3itmdr7i6pv7mpwrqisb.py
# Source Nodes: [x3], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x3 => add_40, clone_32, rsqrt_10, var_mean_10
triton_per_fused_native_layer_norm_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 320.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tizgrwpwh5pa4u3xpnpjb76zmvjpslk64yejmw7h3mjnnjfivm.py
# Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
# x3 => add_40, clone_32, mul_40, rsqrt_10, sub_14, var_mean_10
triton_poi_fused_native_layer_norm_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 320.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clr4lux6wrwb3v7ichi7a2gl65fc3kty5k74mfcuecogztpcipry.py
# Source Nodes: [cat_29], Original ATen: [aten.cat]
# cat_29 => cat_10
triton_poi_fused_cat_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 320) % 197
    x0 = xindex % 320
    x2 = (xindex // 63040)
    x3 = xindex % 63040
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-320) + x3 + (62720*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr3 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h23f4yvs3ptd4imlh4vz55bxguyp2t5gwnfeqvh2mhv4swigv3.py
# Source Nodes: [cat_28], Original ATen: [aten.cat]
# cat_28 => cat_11
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (320*x2) + (63040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*y3) + (((-1) + x2) % 196)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (320 + y0 + (320*(((-1) + x2) % 196)) + (63040*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (320*x2) + (63040*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4aproikobuynfqga4l6dhjsqeecskm5dvmkc3xmbs5oufjdrodd.py
# Source Nodes: [cur_16, l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
# cur_16 => add_43, add_44, mul_42, mul_43, rsqrt_11, sub_15, var_mean_11
# l__mod___serial_blocks3_0_factoratt_crpe_qkv => view_87
triton_per_fused_native_layer_norm_view_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 320, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 320.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (320*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3mzc22cbmerdah53cygdkcgv3qd7cm7fsgymrzdwgx6772q7nm.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => amax_4, clone_33
triton_red_fused__softmax_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4towylkmz5bh7ux2wzzv5vgv55najvuuoev3qeo5mlyr4xmmdos.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => amax_4, clone_33
triton_per_fused__softmax_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceykww2p7ggcssc75vlwgwdmw6imn3bi72ril52yjl6virjgyfsz.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => clone_33, exp_4, sub_16, sum_5
triton_red_fused__softmax_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (320*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqkqxsquxsisvuchrokelzaqvspvmb77egqoacnqdjn5a4pptxl2.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => clone_33, exp_4, sub_16, sum_5
triton_per_fused__softmax_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44uao3y4ic6uhvrrzeddo4wvhz5uo4tcc4mjsj2t3afjka44ijl.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax, aten.detach]
# k_softmax_4 => clone_33, div_4, exp_4, sub_16
triton_poi_fused__softmax_detach_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_detach_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 7880
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 40
    x3 = (xindex // 40)
    y0 = yindex % 8
    y1 = (yindex // 8)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x2 + (40*y0) + (960*x3) + (189120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (40*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (40*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5 + (7880*y4)), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (8*x5) + (63040*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqojptre2ru5fpjj6by4vicx4u5hm7c3bdeh3t6ylvjl3hc4u4i.py
# Source Nodes: [factor_att_8], Original ATen: [aten.clone]
# factor_att_8 => clone_34
triton_poi_fused_clone_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (640 + x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbk2sbb55b6fjzc5vi4g4ycmygjc3526swzjwl3jlgypmgujzwt.py
# Source Nodes: [factor_att_9], Original ATen: [aten.clone]
# factor_att_9 => clone_35
triton_poi_fused_clone_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2s4ldb2h7ftixgqhcuavsv7wbuw7hqojptlpn7vflaopk3uvyi.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_12
triton_poi_fused_cat_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 320
    x2 = xindex
    y1 = (yindex // 320)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (15680*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 200, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + ((-15680) + x2 + (196*y0) + (23520*y1)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1, 1], 320, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((-39200) + x2 + (196*y0) + (23520*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-200) + y0, [XBLOCK, YBLOCK])), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (y0 + (320*x2) + (62720*y1)), tmp28, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebwqr52h63g27bedqdtdzsc45bipcbtl6h3ladgdnvj62o2iwlo.py
# Source Nodes: [x_91], Original ATen: [aten.view]
# x_91 => view_99
triton_poi_fused_view_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 320
    x1 = (xindex // 320)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((40*(x1 % 197)) + (7880*(x0 // 40)) + (63040*(x1 // 197)) + (x0 % 40)), xmask)
    tmp1 = 0.15811388300841897
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + (x1 % 197)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (960*x1)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (320*(((-1) + (x1 % 197)) % 196)) + (62720*(x1 // 197))), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5znkcjedzbjoqo7yzwg27tdkfcxg34mlnfkrr7lfkv4ydhqrdt.py
# Source Nodes: [cur_18, x_93, x_95], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cur_18 => add_47, add_48, mul_46, mul_47, rsqrt_12, sub_17, var_mean_12
# x_93 => add_46
# x_95 => view_101
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 320, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 320.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (320*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43mal5m6t3jm5ha5jpp3othne4kijcv3lu5nuyqhmhaiavoy5kr.py
# Source Nodes: [x_96, x_99], Original ATen: [aten.gelu, aten.view]
# x_96 => add_49, erf_4, mul_48, mul_49, mul_50
# x_99 => view_103
triton_poi_fused_gelu_view_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2017280
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


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pwmilukjh6s7wbthq7jmgpbjviwas3frvi54nltv3ydzef7dhz.py
# Source Nodes: [x3_2, x_93], Original ATen: [aten.add]
# x3_2 => add_50
# x_93 => add_46
triton_poi_fused_add_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctxy2r4o7lxwe7tshqomzawba6jm4cajmzh64m4sqsog7rxpzd3.py
# Source Nodes: [cat_26], Original ATen: [aten.cat]
# cat_26 => cat_13
triton_poi_fused_cat_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0 + (63040*y1), [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*y3) + (((-1) + x2) % 196)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (320 + y0 + (320*(((-1) + x2) % 196)) + (63040*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (320*x2) + (63040*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cue7z6rtmsysvtxnxedsvyqeeetmoskt5vzijevaz5n3son6r5zi.py
# Source Nodes: [x3_nocls], Original ATen: [aten.clone]
# x3_nocls => clone_47
triton_poi_fused_clone_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 62720)
    x3 = xindex % 62720
    x0 = xindex % 320
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x3 + (63040*x2)), None)
    tmp1 = tl.load(in_ptr1 + (320 + x3 + (63040*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (320 + x3 + (63040*x2)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdey66wtewg3fl25qj4icikfxw66lt4qqzy45nif6iqypnlde2d.py
# Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
# x4 => clone_48, var_mean_15
triton_red_fused_native_layer_norm_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r3) + (6272*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznqm6zda7y4gpawliaqtlgho7ix5nsanmr6pbrfzmqrpqlv5rix.py
# Source Nodes: [x4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x4 => add_60, clone_48, rsqrt_15, var_mean_15
triton_per_fused_native_layer_norm_native_layer_norm_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
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
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce65agdc5klzyb4lazaq7cgs5wj3opqchqay7vqzystnccjooldd.py
# Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
# x4 => add_60, clone_48, mul_60, rsqrt_15, sub_21, var_mean_15
triton_poi_fused_native_layer_norm_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cert6k2lkauqi5issahcgj5ywkthn6fkgwvmcuur6dracqw55wlt.py
# Source Nodes: [cat_24], Original ATen: [aten.cat]
# cat_24 => cat_15
triton_poi_fused_cat_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 50
    x0 = xindex % 512
    x2 = (xindex // 25600)
    x3 = xindex % 25600
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-512) + x3 + (25088*x2)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr3 + (x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (x4), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpb5b4iwz5arqd6dk4pewi6ma5oyuxp6ujfw7imd4nhcxzjtnj2d.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat_16
triton_poi_fused_cat_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 50
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (512*x2) + (25600*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*y3) + (((-1) + x2) % 49)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (512 + y0 + (512*(((-1) + x2) % 49)) + (25600*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (512*x2) + (25600*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5xacvirb3moyz765ftidbruays5jngpj76gsv5pqugucx2jhkq.py
# Source Nodes: [cur_24, l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
# cur_24 => add_63, add_64, mul_62, mul_63, rsqrt_16, sub_22, var_mean_16
# l__mod___serial_blocks4_0_factoratt_crpe_qkv => view_129
triton_per_fused_native_layer_norm_view_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_66', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuz27mcyk3hy74zk4fsb7flyzeixv3was4qtxexrpjsbvnubjcn3.py
# Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
# k_softmax_6 => amax_6, clone_49, exp_6, sub_23, sum_7
triton_per_fused__softmax_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*r2) + (76800*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chadex3u47dgb3wmosois35xzw7zjka5smwsx3maethcb3dj4a7f.py
# Source Nodes: [k_softmax_6], Original ATen: [aten._softmax, aten.detach]
# k_softmax_6 => clone_49, div_6, exp_6, sub_23
triton_poi_fused__softmax_detach_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_detach_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 3200
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 8
    y1 = (yindex // 8)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x2 + (64*y0) + (1536*x3) + (76800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (64*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5 + (3200*y4)), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (8*x5) + (25600*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7yabyn4rhufl6hu7ncfl7rxeobulhrk2dsonz3y77zy65f22dg.py
# Source Nodes: [factor_att_12], Original ATen: [aten.clone]
# factor_att_12 => clone_50
triton_poi_fused_clone_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6t64q4ddtvrgdyu74vhjptgtycb6hmmgio6a43tpquictg4ohq2.py
# Source Nodes: [factor_att_13], Original ATen: [aten.clone]
# factor_att_13 => clone_51
triton_poi_fused_clone_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnn3wthjjy72nkvsu7a6dswtoru3lizziox4xiw6tgrjwb5hhzy.py
# Source Nodes: [cat_22], Original ATen: [aten.cat]
# cat_22 => cat_17
triton_poi_fused_cat_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 49
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
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (6272*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 320, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + ((-6272) + x2 + (49*y0) + (9408*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1, 1], 512, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((-15680) + x2 + (49*y0) + (9408*y1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-320) + y0, [XBLOCK, YBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (y0 + (512*x2) + (25088*y1)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cveqkrd4xx74b62xwpfawba2edgdoz6bsmggpju3kvwv6f67jw2y.py
# Source Nodes: [x_131], Original ATen: [aten.view]
# x_131 => view_141
triton_poi_fused_view_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 50)) + (3200*(x0 // 64)) + (25600*(x1 // 50)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + (x1 % 50)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (1536*x1)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (512*(((-1) + (x1 % 50)) % 49)) + (25088*(x1 // 50))), tmp5, other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsxhmqxxu2gs6kylya5wyhy7255tisnzlgwcdki3oi25zgtjvut.py
# Source Nodes: [cur_26, x_133, x_135], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cur_26 => add_67, add_68, mul_66, mul_67, rsqrt_17, sub_24, var_mean_17
# x_133 => add_66
# x_135 => view_143
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cs/ccsw3qxfo7b7skgbfuh22k25r2ot57kc5f4r5cnajrpqu7pkx4po.py
# Source Nodes: [x_136, x_139], Original ATen: [aten.gelu, aten.view]
# x_136 => add_69, erf_6, mul_68, mul_69, mul_70
# x_139 => view_145
triton_poi_fused_gelu_view_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rpz4b7ba77k4pal33jjex24bojukdrw6wztmwxh5ipbd5b32lt.py
# Source Nodes: [x4_2, x_133], Original ATen: [aten.add]
# x4_2 => add_70
# x_133 => add_66
triton_poi_fused_add_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciimh6ffguqd2yulnlpfbud7mtkpj2nycrxslahk6dnqmn5kq6sv.py
# Source Nodes: [cat_21], Original ATen: [aten.cat]
# cat_21 => cat_18
triton_poi_fused_cat_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 50
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0 + (25600*y1), [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*y3) + (((-1) + x2) % 49)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (512 + y0 + (512*(((-1) + x2) % 49)) + (25600*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (512*x2) + (25600*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6ni7wt7tmgfoknpaml6uud6vrtfg3hfend653uu6ew4dlrtaby6.py
# Source Nodes: [x4_3, x_151, x_feat], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x4_3 => add_79
# x_151 => add_75
# x_feat => add_80, mul_80, rsqrt_20, sub_28, var_mean_20
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcul34kzlkdbzwhjjoqk4dkmy3t4ftysxugczjwj5bzdpvubsvi.py
# Source Nodes: [x_162], Original ATen: [aten.clone]
# x_162 => clone_64
triton_poi_fused_clone_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25600*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 64), (64, 64, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (1, 1, 320), (320, 320, 1))
    assert_size_stride(primals_20, (320, ), (1, ))
    assert_size_stride(primals_21, (320, ), (1, ))
    assert_size_stride(primals_22, (320, ), (1, ))
    assert_size_stride(primals_23, (320, ), (1, ))
    assert_size_stride(primals_24, (320, ), (1, ))
    assert_size_stride(primals_25, (320, ), (1, ))
    assert_size_stride(primals_26, (320, ), (1, ))
    assert_size_stride(primals_27, (320, ), (1, ))
    assert_size_stride(primals_28, (1, 1, 512), (512, 512, 1))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (192, 64), (64, 1))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (24, ), (1, ))
    assert_size_stride(primals_51, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_52, (24, ), (1, ))
    assert_size_stride(primals_53, (64, 64), (64, 1))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (512, 64), (64, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (64, 512), (512, 1))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (192, 64), (64, 1))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (64, 64), (64, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (512, 64), (64, 1))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (64, 512), (512, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (384, 128), (128, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_78, (48, ), (1, ))
    assert_size_stride(primals_79, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_80, (48, ), (1, ))
    assert_size_stride(primals_81, (128, 128), (128, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (1024, 128), (128, 1))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (128, 1024), (1024, 1))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (384, 128), (128, 1))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (128, 128), (128, 1))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (1024, 128), (128, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (128, 1024), (1024, 1))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_96, (320, ), (1, ))
    assert_size_stride(primals_97, (320, ), (1, ))
    assert_size_stride(primals_98, (320, ), (1, ))
    assert_size_stride(primals_99, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_101, (960, 320), (320, 1))
    assert_size_stride(primals_102, (960, ), (1, ))
    assert_size_stride(primals_103, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (80, ), (1, ))
    assert_size_stride(primals_105, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (120, ), (1, ))
    assert_size_stride(primals_107, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_108, (120, ), (1, ))
    assert_size_stride(primals_109, (320, 320), (320, 1))
    assert_size_stride(primals_110, (320, ), (1, ))
    assert_size_stride(primals_111, (1280, 320), (320, 1))
    assert_size_stride(primals_112, (1280, ), (1, ))
    assert_size_stride(primals_113, (320, 1280), (1280, 1))
    assert_size_stride(primals_114, (320, ), (1, ))
    assert_size_stride(primals_115, (960, 320), (320, 1))
    assert_size_stride(primals_116, (960, ), (1, ))
    assert_size_stride(primals_117, (320, 320), (320, 1))
    assert_size_stride(primals_118, (320, ), (1, ))
    assert_size_stride(primals_119, (1280, 320), (320, 1))
    assert_size_stride(primals_120, (1280, ), (1, ))
    assert_size_stride(primals_121, (320, 1280), (1280, 1))
    assert_size_stride(primals_122, (320, ), (1, ))
    assert_size_stride(primals_123, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (1536, 512), (512, 1))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_131, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_136, (192, ), (1, ))
    assert_size_stride(primals_137, (512, 512), (512, 1))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (2048, 512), (512, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (512, 2048), (2048, 1))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (1536, 512), (512, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (512, 512), (512, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (2048, 512), (512, 1))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_149, (512, 2048), (2048, 1))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (1000, 512), (512, 1))
    assert_size_stride(primals_152, (1000, ), (1, ))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_39, buf0, 192, 16, grid=grid(192, 16), stream=stream0)
        del primals_39
        buf1 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_67, buf1, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del primals_67
        buf2 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_95, buf2, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del primals_95
        buf3 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_123, buf3, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del primals_123
        buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_153, buf4, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_153
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, buf0, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf9 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf313 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_5.run(buf5, primals_40, buf9, buf313, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_40
        buf10 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(primals_1, buf9, primals_41, primals_42, buf10, 1606144, grid=grid(1606144), stream=stream0)
        del primals_1
        del primals_42
        # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(reinterpret_tensor(buf10, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf11, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf12 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf10, buf11, primals_44, buf12, 512, 3137, grid=grid(512, 3137), stream=stream0)
        buf13 = empty((8, 3137, 1), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cuda', dtype=torch.float32)
        buf16 = reinterpret_tensor(buf14, (8, 3137, 1), (3137, 1, 1), 0); del buf14  # reuse
        buf17 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur, l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_8.run(buf16, buf12, primals_2, primals_3, buf13, buf17, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_3
        buf18 = empty((25096, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_46, buf17, reinterpret_tensor(primals_45, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf18)
        del primals_46
        buf19 = empty_strided((8, 8, 1, 8, 25), (1600, 8, 12800, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf18, buf19, 12800, 126, grid=grid(12800), stream=stream0)
        buf20 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf19, buf20, 512, 25, grid=grid(512), stream=stream0)
        buf21 = buf19; del buf19  # reuse
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_11.run(buf18, buf20, buf21, 12800, 126, grid=grid(12800), stream=stream0)
        buf22 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_12.run(buf21, buf22, 512, 25, grid=grid(512), stream=stream0)
        buf23 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        buf312 = empty_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_13.run(buf18, buf20, buf22, buf23, buf312, 64, 25096, grid=grid(64, 25096), stream=stream0)
        buf24 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf18, buf24, 1606144, grid=grid(1606144), stream=stream0)
        buf25 = empty((64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf24, (64, 3137, 8), (25096, 8, 1), 0), out=buf25)
        buf26 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf18, buf26, 1606144, grid=grid(1606144), stream=stream0)
        buf27 = empty((64, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf25, (64, 8, 8), (64, 8, 1), 0), out=buf27)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf28, (8, 16, 56, 56), (50176, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), primals_49, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf29, (8, 24, 56, 56), (75264, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), primals_51, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf30, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf31 = reinterpret_tensor(buf11, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf11  # reuse
        # Source Nodes: [cat_37], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf28, primals_48, buf29, primals_50, buf30, primals_52, buf31, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del buf28
        del buf29
        del buf30
        buf32 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.view]
        triton_poi_fused_view_17.run(buf27, buf18, buf31, buf32, 1606144, grid=grid(1606144), stream=stream0)
        buf33 = reinterpret_tensor(buf27, (25096, 64), (64, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf32, reinterpret_tensor(primals_53, (64, 64), (1, 64), 0), out=buf33)
        buf37 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        buf38 = empty((25096, 64), device='cuda', dtype=torch.float32)
        buf311 = empty((8, 3137, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_2, x_13, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18.run(buf12, buf33, primals_54, primals_4, primals_5, buf37, buf38, buf311, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_5
        buf39 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_56, buf38, reinterpret_tensor(primals_55, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf39)
        del primals_56
        buf40 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16, x_19], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_19.run(buf39, buf40, 12849152, grid=grid(12849152), stream=stream0)
        buf41 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_57, (512, 64), (1, 512), 0), out=buf41)
        buf42 = reinterpret_tensor(buf41, (8, 3137, 64), (200768, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [x1_2, x_13], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf42, buf12, buf33, primals_54, primals_58, 1606144, grid=grid(1606144), stream=stream0)
        del primals_54
        del primals_58
        # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf43, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf44 = reinterpret_tensor(buf33, (8, 3137, 64), (200768, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [cat_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(buf42, buf43, primals_44, buf44, 512, 3137, grid=grid(512, 3137), stream=stream0)
        del primals_44
        buf45 = empty((8, 3137, 1), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cuda', dtype=torch.float32)
        buf48 = reinterpret_tensor(buf46, (8, 3137, 1), (3137, 1, 1), 0); del buf46  # reuse
        buf49 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_4, l__mod___serial_blocks1_1_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_8.run(buf48, buf44, primals_6, primals_7, buf45, buf49, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_7
        buf50 = empty((25096, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks1_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf49, reinterpret_tensor(primals_59, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf50)
        del primals_60
        buf51 = buf21; del buf21  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf50, buf51, 12800, 126, grid=grid(12800), stream=stream0)
        buf52 = buf22; del buf22  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf51, buf52, 512, 25, grid=grid(512), stream=stream0)
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_11.run(buf50, buf52, buf53, 12800, 126, grid=grid(12800), stream=stream0)
        buf54 = buf20; del buf20  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_12.run(buf53, buf54, 512, 25, grid=grid(512), stream=stream0)
        del buf53
        buf55 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_13.run(buf50, buf52, buf54, buf55, buf310, 64, 25096, grid=grid(64, 25096), stream=stream0)
        del buf52
        del buf54
        buf56 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf50, buf56, 1606144, grid=grid(1606144), stream=stream0)
        buf57 = empty((64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf56, (64, 3137, 8), (25096, 8, 1), 0), out=buf57)
        buf58 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf50, buf58, 1606144, grid=grid(1606144), stream=stream0)
        buf59 = empty((64, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf57, (64, 8, 8), (64, 8, 1), 0), out=buf59)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(reinterpret_tensor(buf50, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf60, (8, 16, 56, 56), (50176, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(reinterpret_tensor(buf50, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), primals_49, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf61, (8, 24, 56, 56), (75264, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(reinterpret_tensor(buf50, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), primals_51, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf62, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf63 = reinterpret_tensor(buf43, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf43  # reuse
        # Source Nodes: [cat_35], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf60, primals_48, buf61, primals_50, buf62, primals_52, buf63, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del buf60
        del buf61
        del buf62
        del primals_48
        del primals_50
        del primals_52
        buf64 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.view]
        triton_poi_fused_view_17.run(buf59, buf50, buf63, buf64, 1606144, grid=grid(1606144), stream=stream0)
        buf65 = reinterpret_tensor(buf59, (25096, 64), (64, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf64, reinterpret_tensor(primals_61, (64, 64), (1, 64), 0), out=buf65)
        buf69 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        buf70 = empty((25096, 64), device='cuda', dtype=torch.float32)
        buf309 = empty((8, 3137, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_6, x_31, x_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18.run(buf44, buf65, primals_62, primals_8, primals_9, buf69, buf70, buf309, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_9
        buf71 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, buf70, reinterpret_tensor(primals_63, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf71)
        del primals_64
        buf72 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34, x_37], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_19.run(buf71, buf72, 12849152, grid=grid(12849152), stream=stream0)
        buf73 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_65, (512, 64), (1, 512), 0), out=buf73)
        buf74 = reinterpret_tensor(buf5, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf5  # reuse
        # Source Nodes: [x1_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf44, buf65, primals_62, buf73, primals_66, buf74, 1605632, grid=grid(1605632), stream=stream0)
        del buf65
        del buf73
        del primals_62
        del primals_66
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf79 = empty((8, 784, 128), device='cuda', dtype=torch.float32)
        buf308 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x2], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_23.run(buf75, primals_68, buf79, buf308, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_68
        buf80 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(primals_10, buf79, primals_69, primals_70, buf80, 803840, grid=grid(803840), stream=stream0)
        del primals_10
        del primals_70
        # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(reinterpret_tensor(buf80, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), primals_71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf81, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf82 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf80, buf81, primals_72, buf82, 1024, 785, grid=grid(1024, 785), stream=stream0)
        buf83 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        buf86 = reinterpret_tensor(buf84, (8, 785, 1), (785, 1, 1), 0); del buf84  # reuse
        buf87 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_8, l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_26.run(buf86, buf82, primals_11, primals_12, buf83, buf87, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_12
        buf88 = empty((6280, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_74, buf87, reinterpret_tensor(primals_73, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf88)
        del primals_74
        buf89 = empty_strided((8, 8, 1, 16, 7), (896, 16, 7168, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_27.run(buf88, buf89, 7168, 113, grid=grid(7168), stream=stream0)
        buf90 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_28.run(buf89, buf90, 1024, 7, grid=grid(1024), stream=stream0)
        buf91 = buf89; del buf89  # reuse
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_29.run(buf88, buf90, buf91, 7168, 113, grid=grid(7168), stream=stream0)
        buf92 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_30.run(buf91, buf92, 1024, 7, grid=grid(1024), stream=stream0)
        buf93 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_31.run(buf88, buf90, buf92, buf93, buf307, 64, 12560, grid=grid(64, 12560), stream=stream0)
        buf94 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf88, buf94, 803840, grid=grid(803840), stream=stream0)
        buf95 = empty((64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf94, (64, 785, 16), (12560, 16, 1), 0), out=buf95)
        buf96 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf88, buf96, 803840, grid=grid(803840), stream=stream0)
        buf97 = empty((64, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf95, (64, 16, 16), (256, 16, 1), 0), out=buf97)
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf98, (8, 32, 28, 28), (25088, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), primals_77, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf99, (8, 48, 28, 28), (37632, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), primals_79, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf100, (8, 48, 28, 28), (37632, 784, 28, 1))
        buf101 = reinterpret_tensor(buf81, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf81  # reuse
        # Source Nodes: [cat_32], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf98, primals_76, buf99, primals_78, buf100, primals_80, buf101, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del buf100
        del buf98
        del buf99
        buf102 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten.view]
        triton_poi_fused_view_35.run(buf97, buf88, buf101, buf102, 803840, grid=grid(803840), stream=stream0)
        buf103 = reinterpret_tensor(buf97, (6280, 128), (128, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf102, reinterpret_tensor(primals_81, (128, 128), (1, 128), 0), out=buf103)
        buf107 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        buf108 = empty((6280, 128), device='cuda', dtype=torch.float32)
        buf306 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_10, x_53, x_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_36.run(buf82, buf103, primals_82, primals_13, primals_14, buf107, buf108, buf306, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_14
        buf109 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf108, reinterpret_tensor(primals_83, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf109)
        del primals_84
        buf110 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56, x_59], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_37.run(buf109, buf110, 6430720, grid=grid(6430720), stream=stream0)
        buf111 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_85, (1024, 128), (1, 1024), 0), out=buf111)
        buf112 = reinterpret_tensor(buf111, (8, 785, 128), (100480, 128, 1), 0); del buf111  # reuse
        # Source Nodes: [x2_2, x_53], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf112, buf82, buf103, primals_82, primals_86, 803840, grid=grid(803840), stream=stream0)
        del primals_82
        del primals_86
        # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(reinterpret_tensor(buf112, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), primals_71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf113, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf114 = reinterpret_tensor(buf103, (8, 785, 128), (100480, 128, 1), 0); del buf103  # reuse
        # Source Nodes: [cat_31], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf112, buf113, primals_72, buf114, 1024, 785, grid=grid(1024, 785), stream=stream0)
        del primals_72
        buf115 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        buf116 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        buf118 = reinterpret_tensor(buf116, (8, 785, 1), (785, 1, 1), 0); del buf116  # reuse
        buf119 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_12, l__mod___serial_blocks2_1_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_26.run(buf118, buf114, primals_15, primals_16, buf115, buf119, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_16
        buf120 = empty((6280, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks2_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf119, reinterpret_tensor(primals_87, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf120)
        del primals_88
        buf121 = buf91; del buf91  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_27.run(buf120, buf121, 7168, 113, grid=grid(7168), stream=stream0)
        buf122 = buf92; del buf92  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_28.run(buf121, buf122, 1024, 7, grid=grid(1024), stream=stream0)
        buf123 = buf121; del buf121  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_29.run(buf120, buf122, buf123, 7168, 113, grid=grid(7168), stream=stream0)
        buf124 = buf90; del buf90  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_30.run(buf123, buf124, 1024, 7, grid=grid(1024), stream=stream0)
        del buf123
        buf125 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        buf305 = empty_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_31.run(buf120, buf122, buf124, buf125, buf305, 64, 12560, grid=grid(64, 12560), stream=stream0)
        del buf122
        del buf124
        buf126 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf120, buf126, 803840, grid=grid(803840), stream=stream0)
        buf127 = empty((64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf126, (64, 785, 16), (12560, 16, 1), 0), out=buf127)
        buf128 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf120, buf128, 803840, grid=grid(803840), stream=stream0)
        buf129 = empty((64, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf127, (64, 16, 16), (256, 16, 1), 0), out=buf129)
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(reinterpret_tensor(buf120, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf130, (8, 32, 28, 28), (25088, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(reinterpret_tensor(buf120, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), primals_77, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf131, (8, 48, 28, 28), (37632, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(reinterpret_tensor(buf120, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), primals_79, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf132, (8, 48, 28, 28), (37632, 784, 28, 1))
        buf133 = reinterpret_tensor(buf113, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf113  # reuse
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf130, primals_76, buf131, primals_78, buf132, primals_80, buf133, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del buf131
        del buf132
        del primals_76
        del primals_78
        del primals_80
        buf134 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.view]
        triton_poi_fused_view_35.run(buf129, buf120, buf133, buf134, 803840, grid=grid(803840), stream=stream0)
        buf135 = reinterpret_tensor(buf129, (6280, 128), (128, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf134, reinterpret_tensor(primals_89, (128, 128), (1, 128), 0), out=buf135)
        buf139 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        buf140 = empty((6280, 128), device='cuda', dtype=torch.float32)
        buf304 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_14, x_71, x_73], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_36.run(buf114, buf135, primals_90, primals_17, primals_18, buf139, buf140, buf304, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_18
        buf141 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf140, reinterpret_tensor(primals_91, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf141)
        del primals_92
        buf142 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_77], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_37.run(buf141, buf142, 6430720, grid=grid(6430720), stream=stream0)
        buf143 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf142, reinterpret_tensor(primals_93, (1024, 128), (1, 1024), 0), out=buf143)
        buf144 = reinterpret_tensor(buf75, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf75  # reuse
        # Source Nodes: [x2_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf114, buf135, primals_90, buf143, primals_94, buf144, 802816, grid=grid(802816), stream=stream0)
        del buf135
        del buf143
        del primals_90
        del primals_94
        # Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 320, 14, 14), (62720, 196, 14, 1))
        buf146 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf148 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_41.run(buf145, primals_96, buf146, buf147, buf148, 4704, 107, grid=grid(4704), stream=stream0)
        buf149 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf303 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_42.run(buf146, buf147, buf148, buf149, buf150, buf303, 1568, 3, grid=grid(1568), stream=stream0)
        del buf146
        del buf147
        del buf148
        buf152 = empty((8, 196, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_43.run(buf145, primals_96, buf149, buf150, buf152, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del primals_96
        buf153 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_44.run(primals_19, buf152, primals_97, primals_98, buf153, 504320, grid=grid(504320), stream=stream0)
        del primals_19
        del primals_98
        # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(reinterpret_tensor(buf153, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), primals_99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf154, (8, 320, 14, 14), (62720, 196, 14, 1))
        buf155 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf153, buf154, primals_100, buf155, 2560, 197, grid=grid(2560, 197), stream=stream0)
        buf156 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf159 = reinterpret_tensor(buf157, (8, 197, 1), (197, 1, 1), 0); del buf157  # reuse
        buf160 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_16, l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_46.run(buf159, buf155, primals_20, primals_21, buf156, buf160, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_21
        buf161 = empty((1576, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf160, reinterpret_tensor(primals_101, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf161)
        del primals_102
        buf162 = empty_strided((8, 8, 1, 40, 2), (640, 40, 5120, 1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_47.run(buf161, buf162, 5120, 99, grid=grid(5120), stream=stream0)
        buf163 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_48.run(buf162, buf163, 2560, 2, grid=grid(2560), stream=stream0)
        buf164 = buf162; del buf162  # reuse
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_49.run(buf161, buf163, buf164, 5120, 99, grid=grid(5120), stream=stream0)
        buf165 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_50.run(buf164, buf165, 2560, 2, grid=grid(2560), stream=stream0)
        buf166 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        buf302 = empty_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_51.run(buf161, buf163, buf165, buf166, buf302, 64, 7880, grid=grid(64, 7880), stream=stream0)
        buf167 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf161, buf167, 504320, grid=grid(504320), stream=stream0)
        buf168 = empty((64, 40, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf167, (64, 197, 40), (7880, 40, 1), 0), out=buf168)
        buf169 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf161, buf169, 504320, grid=grid(504320), stream=stream0)
        buf170 = empty((64, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf168, (64, 40, 40), (1600, 40, 1), 0), out=buf170)
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(reinterpret_tensor(buf161, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf171, (8, 80, 14, 14), (15680, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(reinterpret_tensor(buf161, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), primals_105, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf172, (8, 120, 14, 14), (23520, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(reinterpret_tensor(buf161, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), primals_107, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf173, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf174 = reinterpret_tensor(buf154, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf154  # reuse
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf171, primals_104, buf172, primals_106, buf173, primals_108, buf174, 2560, 196, grid=grid(2560, 196), stream=stream0)
        del buf171
        del buf172
        del buf173
        buf175 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.view]
        triton_poi_fused_view_55.run(buf170, buf161, buf174, buf175, 504320, grid=grid(504320), stream=stream0)
        buf176 = reinterpret_tensor(buf170, (1576, 320), (320, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf175, reinterpret_tensor(primals_109, (320, 320), (1, 320), 0), out=buf176)
        buf180 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        buf181 = empty((1576, 320), device='cuda', dtype=torch.float32)
        buf301 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_18, x_93, x_95], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_56.run(buf155, buf176, primals_110, primals_22, primals_23, buf180, buf181, buf301, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_23
        buf182 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_112, buf181, reinterpret_tensor(primals_111, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf182)
        del primals_112
        buf183 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_57.run(buf182, buf183, 2017280, grid=grid(2017280), stream=stream0)
        buf184 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_113, (1280, 320), (1, 1280), 0), out=buf184)
        buf185 = reinterpret_tensor(buf184, (8, 197, 320), (63040, 320, 1), 0); del buf184  # reuse
        # Source Nodes: [x3_2, x_93], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf185, buf155, buf176, primals_110, primals_114, 504320, grid=grid(504320), stream=stream0)
        del primals_110
        del primals_114
        # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(reinterpret_tensor(buf185, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), primals_99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf186, (8, 320, 14, 14), (62720, 196, 14, 1))
        buf187 = reinterpret_tensor(buf176, (8, 197, 320), (63040, 320, 1), 0); del buf176  # reuse
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf185, buf186, primals_100, buf187, 2560, 197, grid=grid(2560, 197), stream=stream0)
        del primals_100
        buf188 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf189 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf191 = reinterpret_tensor(buf189, (8, 197, 1), (197, 1, 1), 0); del buf189  # reuse
        buf192 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_20, l__mod___serial_blocks3_1_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_46.run(buf191, buf187, primals_24, primals_25, buf188, buf192, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_25
        buf193 = empty((1576, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks3_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf192, reinterpret_tensor(primals_115, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf193)
        del primals_116
        buf194 = buf164; del buf164  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_47.run(buf193, buf194, 5120, 99, grid=grid(5120), stream=stream0)
        buf195 = buf165; del buf165  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_48.run(buf194, buf195, 2560, 2, grid=grid(2560), stream=stream0)
        buf196 = buf194; del buf194  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_49.run(buf193, buf195, buf196, 5120, 99, grid=grid(5120), stream=stream0)
        buf197 = buf163; del buf163  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_50.run(buf196, buf197, 2560, 2, grid=grid(2560), stream=stream0)
        del buf196
        buf198 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        buf300 = empty_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_51.run(buf193, buf195, buf197, buf198, buf300, 64, 7880, grid=grid(64, 7880), stream=stream0)
        del buf195
        del buf197
        buf199 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf193, buf199, 504320, grid=grid(504320), stream=stream0)
        buf200 = empty((64, 40, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf199, (64, 197, 40), (7880, 40, 1), 0), out=buf200)
        buf201 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf193, buf201, 504320, grid=grid(504320), stream=stream0)
        buf202 = empty((64, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf200, (64, 40, 40), (1600, 40, 1), 0), out=buf202)
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(reinterpret_tensor(buf193, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf203, (8, 80, 14, 14), (15680, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(reinterpret_tensor(buf193, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), primals_105, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf204, (8, 120, 14, 14), (23520, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(reinterpret_tensor(buf193, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), primals_107, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf205, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf206 = reinterpret_tensor(buf186, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf186  # reuse
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf203, primals_104, buf204, primals_106, buf205, primals_108, buf206, 2560, 196, grid=grid(2560, 196), stream=stream0)
        del buf203
        del buf204
        del buf205
        del primals_104
        del primals_106
        del primals_108
        buf207 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.view]
        triton_poi_fused_view_55.run(buf202, buf193, buf206, buf207, 504320, grid=grid(504320), stream=stream0)
        buf208 = reinterpret_tensor(buf202, (1576, 320), (320, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf207, reinterpret_tensor(primals_117, (320, 320), (1, 320), 0), out=buf208)
        buf212 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        buf213 = empty((1576, 320), device='cuda', dtype=torch.float32)
        buf299 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_22, x_111, x_113], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_56.run(buf187, buf208, primals_118, primals_26, primals_27, buf212, buf213, buf299, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_27
        buf214 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf213, reinterpret_tensor(primals_119, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf214)
        del primals_120
        buf215 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114, x_117], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_57.run(buf214, buf215, 2017280, grid=grid(2017280), stream=stream0)
        buf216 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf215, reinterpret_tensor(primals_121, (1280, 320), (1, 1280), 0), out=buf216)
        buf217 = reinterpret_tensor(buf145, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf145  # reuse
        # Source Nodes: [x3_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf187, buf208, primals_118, buf216, primals_122, buf217, 501760, grid=grid(501760), stream=stream0)
        del buf208
        del buf216
        del primals_118
        del primals_122
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, buf3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf219 = reinterpret_tensor(buf150, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf150  # reuse
        buf220 = reinterpret_tensor(buf149, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf149  # reuse
        buf221 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_61.run(buf218, primals_124, buf219, buf220, buf221, 1568, 128, grid=grid(1568), stream=stream0)
        buf222 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf223 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf298 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_62.run(buf219, buf220, buf221, buf222, buf223, buf298, 392, 4, grid=grid(392), stream=stream0)
        del buf219
        del buf220
        del buf221
        buf225 = reinterpret_tensor(buf130, (8, 49, 512), (25088, 512, 1), 0); del buf130  # reuse
        # Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_63.run(buf218, primals_124, buf222, buf223, buf225, 392, 512, grid=grid(392, 512), stream=stream0)
        del buf218
        del buf222
        del buf223
        del primals_124
        buf226 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(primals_28, buf225, primals_125, primals_126, buf226, 204800, grid=grid(204800), stream=stream0)
        del primals_126
        del primals_28
        # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(reinterpret_tensor(buf226, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf227, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf228 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_65.run(buf226, buf227, primals_128, buf228, 4096, 50, grid=grid(4096, 50), stream=stream0)
        buf229 = empty((8, 50, 1), device='cuda', dtype=torch.float32)
        buf230 = empty_strided((8, 50, 1), (50, 1, 400), device='cuda', dtype=torch.float32)
        buf232 = reinterpret_tensor(buf230, (8, 50, 1), (50, 1, 1), 0); del buf230  # reuse
        buf233 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_24, l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_66.run(buf232, buf228, primals_29, primals_30, buf229, buf233, 400, 512, grid=grid(400), stream=stream0)
        del primals_30
        buf234 = empty((400, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf233, reinterpret_tensor(primals_129, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf234)
        del primals_130
        buf235 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cuda', dtype=torch.float32)
        buf236 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
        triton_per_fused__softmax_67.run(buf234, buf235, buf236, 4096, 50, grid=grid(4096), stream=stream0)
        buf237 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        buf297 = empty_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_6], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_68.run(buf234, buf235, buf236, buf237, buf297, 64, 3200, grid=grid(64, 3200), stream=stream0)
        buf238 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_69.run(buf234, buf238, 204800, grid=grid(204800), stream=stream0)
        buf239 = empty((64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf238, (64, 50, 64), (3200, 64, 1), 0), out=buf239)
        buf240 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_70.run(buf234, buf240, 204800, grid=grid(204800), stream=stream0)
        buf241 = empty((64, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf239, (64, 64, 64), (4096, 64, 1), 0), out=buf241)
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(reinterpret_tensor(buf234, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), primals_131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf242, (8, 128, 7, 7), (6272, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(reinterpret_tensor(buf234, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), primals_133, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf243, (8, 192, 7, 7), (9408, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(reinterpret_tensor(buf234, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), primals_135, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf244, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf245 = reinterpret_tensor(buf227, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf227  # reuse
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf242, primals_132, buf243, primals_134, buf244, primals_136, buf245, 4096, 49, grid=grid(4096, 49), stream=stream0)
        del buf242
        del buf243
        del buf244
        buf246 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131], Original ATen: [aten.view]
        triton_poi_fused_view_72.run(buf241, buf234, buf245, buf246, 204800, grid=grid(204800), stream=stream0)
        buf247 = reinterpret_tensor(buf241, (400, 512), (512, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf246, reinterpret_tensor(primals_137, (512, 512), (1, 512), 0), out=buf247)
        buf251 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        buf252 = empty((400, 512), device='cuda', dtype=torch.float32)
        buf296 = empty((8, 50, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_26, x_133, x_135], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_73.run(buf228, buf247, primals_138, primals_31, primals_32, buf251, buf252, buf296, 400, 512, grid=grid(400), stream=stream0)
        del primals_32
        buf253 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_140, buf252, reinterpret_tensor(primals_139, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf253)
        del primals_140
        buf254 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136, x_139], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_74.run(buf253, buf254, 819200, grid=grid(819200), stream=stream0)
        buf255 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf254, reinterpret_tensor(primals_141, (2048, 512), (1, 2048), 0), out=buf255)
        buf256 = reinterpret_tensor(buf255, (8, 50, 512), (25600, 512, 1), 0); del buf255  # reuse
        # Source Nodes: [x4_2, x_133], Original ATen: [aten.add]
        triton_poi_fused_add_75.run(buf256, buf228, buf247, primals_138, primals_142, 204800, grid=grid(204800), stream=stream0)
        del primals_138
        del primals_142
        # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(reinterpret_tensor(buf256, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf257, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf258 = reinterpret_tensor(buf247, (8, 50, 512), (25600, 512, 1), 0); del buf247  # reuse
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf256, buf257, primals_128, buf258, 4096, 50, grid=grid(4096, 50), stream=stream0)
        del primals_128
        buf259 = empty((8, 50, 1), device='cuda', dtype=torch.float32)
        buf260 = empty_strided((8, 50, 1), (50, 1, 400), device='cuda', dtype=torch.float32)
        buf262 = reinterpret_tensor(buf260, (8, 50, 1), (50, 1, 1), 0); del buf260  # reuse
        buf263 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_28, l__mod___serial_blocks4_1_factoratt_crpe_qkv], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_66.run(buf262, buf258, primals_33, primals_34, buf259, buf263, 400, 512, grid=grid(400), stream=stream0)
        del primals_34
        buf264 = empty((400, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks4_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf263, reinterpret_tensor(primals_143, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf264)
        del primals_144
        buf265 = buf236; del buf236  # reuse
        buf266 = buf235; del buf235  # reuse
        # Source Nodes: [k_softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_67.run(buf264, buf265, buf266, 4096, 50, grid=grid(4096), stream=stream0)
        buf267 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        buf295 = empty_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_7], Original ATen: [aten._softmax, aten.detach]
        triton_poi_fused__softmax_detach_68.run(buf264, buf265, buf266, buf267, buf295, 64, 3200, grid=grid(64, 3200), stream=stream0)
        del buf265
        buf268 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_69.run(buf264, buf268, 204800, grid=grid(204800), stream=stream0)
        buf269 = empty((64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf268, (64, 50, 64), (3200, 64, 1), 0), out=buf269)
        buf270 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_70.run(buf264, buf270, 204800, grid=grid(204800), stream=stream0)
        buf271 = empty((64, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf269, (64, 64, 64), (4096, 64, 1), 0), out=buf271)
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(reinterpret_tensor(buf264, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), primals_131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf272, (8, 128, 7, 7), (6272, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(reinterpret_tensor(buf264, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), primals_133, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf273, (8, 192, 7, 7), (9408, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(reinterpret_tensor(buf264, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), primals_135, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf274, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf275 = reinterpret_tensor(buf257, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf257  # reuse
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf272, primals_132, buf273, primals_134, buf274, primals_136, buf275, 4096, 49, grid=grid(4096, 49), stream=stream0)
        del buf272
        del buf273
        del buf274
        del primals_132
        del primals_134
        del primals_136
        buf276 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.view]
        triton_poi_fused_view_72.run(buf271, buf264, buf275, buf276, 204800, grid=grid(204800), stream=stream0)
        buf277 = reinterpret_tensor(buf271, (400, 512), (512, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf276, reinterpret_tensor(primals_145, (512, 512), (1, 512), 0), out=buf277)
        buf281 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        buf282 = empty((400, 512), device='cuda', dtype=torch.float32)
        buf294 = empty((8, 50, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_30, x_151, x_153], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_73.run(buf258, buf277, primals_146, primals_35, primals_36, buf281, buf282, buf294, 400, 512, grid=grid(400), stream=stream0)
        del primals_36
        buf283 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, buf282, reinterpret_tensor(primals_147, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf283)
        del primals_148
        buf284 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154, x_157], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_74.run(buf283, buf284, 819200, grid=grid(819200), stream=stream0)
        buf285 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf284, reinterpret_tensor(primals_149, (2048, 512), (1, 2048), 0), out=buf285)
        buf286 = reinterpret_tensor(buf285, (8, 50, 512), (25600, 512, 1), 0); del buf285  # reuse
        buf290 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        buf293 = empty((8, 50, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x4_3, x_151, x_feat], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77.run(buf286, buf258, buf277, primals_146, primals_150, buf290, buf293, 400, 512, grid=grid(400), stream=stream0)
        del buf277
        del buf286
        del primals_146
        del primals_150
        buf291 = reinterpret_tensor(buf266, (8, 512), (512, 1), 0); del buf266  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf290, primals_37, primals_38, buf291, 4096, grid=grid(4096), stream=stream0)
        del primals_38
        buf292 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf291, reinterpret_tensor(primals_151, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf292)
        del primals_152
        return (buf292, primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, buf0, primals_41, primals_43, primals_47, primals_49, primals_51, buf1, primals_69, primals_71, primals_75, primals_77, primals_79, buf2, primals_97, primals_99, primals_103, primals_105, primals_107, buf3, primals_125, primals_127, primals_131, primals_133, primals_135, buf4, buf9, reinterpret_tensor(buf10, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), buf12, buf13, buf16, buf17, reinterpret_tensor(buf18, (8, 8, 3136, 8), (602304, 8, 192, 1), 192), reinterpret_tensor(buf18, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), buf31, buf32, buf37, buf38, buf39, buf40, reinterpret_tensor(buf42, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), buf44, buf45, buf48, buf49, reinterpret_tensor(buf50, (8, 8, 3136, 8), (602304, 8, 192, 1), 192), reinterpret_tensor(buf50, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), reinterpret_tensor(buf50, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), reinterpret_tensor(buf50, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), buf63, buf64, buf69, buf70, buf71, buf72, buf74, buf79, reinterpret_tensor(buf80, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), buf82, buf83, buf86, buf87, reinterpret_tensor(buf88, (8, 8, 784, 16), (301440, 16, 384, 1), 384), reinterpret_tensor(buf88, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), reinterpret_tensor(buf88, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), reinterpret_tensor(buf88, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), buf101, buf102, buf107, buf108, buf109, buf110, reinterpret_tensor(buf112, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), buf114, buf115, buf118, buf119, reinterpret_tensor(buf120, (8, 8, 784, 16), (301440, 16, 384, 1), 384), reinterpret_tensor(buf120, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), reinterpret_tensor(buf120, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), reinterpret_tensor(buf120, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), buf133, buf134, buf139, buf140, buf141, buf142, buf144, buf152, reinterpret_tensor(buf153, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), buf155, buf156, buf159, buf160, reinterpret_tensor(buf161, (8, 8, 196, 40), (189120, 40, 960, 1), 960), reinterpret_tensor(buf161, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), reinterpret_tensor(buf161, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), reinterpret_tensor(buf161, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), buf174, buf175, buf180, buf181, buf182, buf183, reinterpret_tensor(buf185, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), buf187, buf188, buf191, buf192, reinterpret_tensor(buf193, (8, 8, 196, 40), (189120, 40, 960, 1), 960), reinterpret_tensor(buf193, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), reinterpret_tensor(buf193, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), reinterpret_tensor(buf193, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), buf206, buf207, buf212, buf213, buf214, buf215, buf217, buf225, reinterpret_tensor(buf226, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), buf228, buf229, buf232, buf233, reinterpret_tensor(buf234, (8, 8, 49, 64), (76800, 64, 1536, 1), 1536), reinterpret_tensor(buf234, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), reinterpret_tensor(buf234, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), reinterpret_tensor(buf234, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), buf245, buf246, buf251, buf252, buf253, buf254, reinterpret_tensor(buf256, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), buf258, buf259, buf262, buf263, reinterpret_tensor(buf264, (8, 8, 49, 64), (76800, 64, 1536, 1), 1536), reinterpret_tensor(buf264, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), reinterpret_tensor(buf264, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), reinterpret_tensor(buf264, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), buf275, buf276, buf281, buf282, buf283, buf284, buf290, buf291, reinterpret_tensor(primals_151, (1000, 512), (512, 1), 0), buf293, reinterpret_tensor(primals_149, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_147, (2048, 512), (512, 1), 0), buf294, reinterpret_tensor(primals_145, (512, 512), (512, 1), 0), reinterpret_tensor(buf270, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf269, (64, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf267, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf268, (64, 64, 50), (3200, 1, 64), 0), buf295, reinterpret_tensor(primals_143, (1536, 512), (512, 1), 0), reinterpret_tensor(primals_141, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_139, (2048, 512), (512, 1), 0), buf296, reinterpret_tensor(primals_137, (512, 512), (512, 1), 0), reinterpret_tensor(buf240, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf239, (64, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf237, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf238, (64, 64, 50), (3200, 1, 64), 0), buf297, reinterpret_tensor(primals_129, (1536, 512), (512, 1), 0), buf298, reinterpret_tensor(primals_121, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_119, (1280, 320), (320, 1), 0), buf299, reinterpret_tensor(primals_117, (320, 320), (320, 1), 0), reinterpret_tensor(buf201, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf200, (64, 40, 40), (1600, 1, 40), 0), reinterpret_tensor(buf198, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf199, (64, 40, 197), (7880, 1, 40), 0), buf300, reinterpret_tensor(primals_115, (960, 320), (320, 1), 0), reinterpret_tensor(primals_113, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_111, (1280, 320), (320, 1), 0), buf301, reinterpret_tensor(primals_109, (320, 320), (320, 1), 0), reinterpret_tensor(buf169, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf168, (64, 40, 40), (1600, 1, 40), 0), reinterpret_tensor(buf166, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf167, (64, 40, 197), (7880, 1, 40), 0), buf302, reinterpret_tensor(primals_101, (960, 320), (320, 1), 0), buf303, reinterpret_tensor(primals_93, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_91, (1024, 128), (128, 1), 0), buf304, reinterpret_tensor(primals_89, (128, 128), (128, 1), 0), reinterpret_tensor(buf128, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf127, (64, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf125, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf126, (64, 16, 785), (12560, 1, 16), 0), buf305, reinterpret_tensor(primals_87, (384, 128), (128, 1), 0), reinterpret_tensor(primals_85, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_83, (1024, 128), (128, 1), 0), buf306, reinterpret_tensor(primals_81, (128, 128), (128, 1), 0), reinterpret_tensor(buf96, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf95, (64, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf93, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf94, (64, 16, 785), (12560, 1, 16), 0), buf307, reinterpret_tensor(primals_73, (384, 128), (128, 1), 0), buf308, reinterpret_tensor(primals_65, (64, 512), (512, 1), 0), reinterpret_tensor(primals_63, (512, 64), (64, 1), 0), buf309, reinterpret_tensor(primals_61, (64, 64), (64, 1), 0), reinterpret_tensor(buf58, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf57, (64, 8, 8), (64, 1, 8), 0), reinterpret_tensor(buf55, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf56, (64, 8, 3137), (25096, 1, 8), 0), buf310, reinterpret_tensor(primals_59, (192, 64), (64, 1), 0), reinterpret_tensor(primals_57, (64, 512), (512, 1), 0), reinterpret_tensor(primals_55, (512, 64), (64, 1), 0), buf311, reinterpret_tensor(primals_53, (64, 64), (64, 1), 0), reinterpret_tensor(buf26, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf25, (64, 8, 8), (64, 1, 8), 0), reinterpret_tensor(buf23, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf24, (64, 8, 3137), (25096, 1, 8), 0), buf312, reinterpret_tensor(primals_45, (192, 64), (64, 1), 0), buf313, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1, 1, 320), (320, 320, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
