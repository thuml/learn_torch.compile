
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqjafhqtlts2rljhwqeypkchbqt5sjdxgwfqoi77pctord4ihgl.py
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
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (4096*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvw4hubdjqcvxi4bu4nbciaj75ero2l77nb2txkz5hmae5pqjjm.py
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
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mg/cmguvpbhk6tsueigdkfpvrg25s3qbxvqsxnvs5htek6iznwtd4xq.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (2048*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gsekc6lv2jcloii2i5zu3ysz6kh2sb7zzajnb6ifmil2dfim2r.py
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
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vi/cvibro3w445ljykzhigjj4ovjgjmjbyuvsusr7rdni64hpp3nwu6.py
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
    size_hints=[131072, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cghpnfqlkhyg25dvfcsgmtersluw7whuaquqfcpvgcvsxtay62cc.py
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
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ow/cowuza6omhjwqiwk73dufeqho4pg2y4ooovjselmkzhdlbho65to.py
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
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd337wtbtlsjmbd3qom6v75vqa4sqdr4bpha2ahjlke5cds3o5aj.py
# Source Nodes: [l__mod___blocks_0_0_attn_q, l__mod___blocks_0_0_norm1, x_2, x_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_0_attn_q => view_1
# l__mod___blocks_0_0_norm1 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_2 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
# x_4 => view_4
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp36 / tmp11
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(rmask & xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = tmp29 - tmp37
    tmp45 = tmp43 / tmp20
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp44 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tmp48 / tmp20
    tmp55 = tmp24 / tmp20
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp25, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (64*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr6 + (r2 + (64*x3)), tmp53, rmask & xmask)
    tl.store(out_ptr7 + (r2 + (64*x3)), tmp53, rmask & xmask)
    tl.store(out_ptr8 + (x3), tmp54, xmask)
    tl.store(out_ptr9 + (x3), tmp55, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvm2qhl2z5uvsfn55wwpfk7x3pfv3gqapxcgkxp2ylc5hhc6ybe.py
# Source Nodes: [l__mod___blocks_0_0_attn_kv, x_6], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_0_attn_kv => view_6
# x_6 => add_4, add_5, clone_2, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (3136*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3yvdp2jpgjwvdr3ovwlwyufuxpq3kbl5o7plh4h5p23qiocfrn.py
# Source Nodes: [l__mod___blocks_0_0_norm2, x_11, x_12, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_0_norm2 => add_7, add_8, mul_6, mul_7, rsqrt_3, sub_3, var_mean_3
# x_11 => add_6
# x_12 => view_12
# x_2 => add_1, mul_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccv7khulexejhya4qptvyq4zbi6jaazhluevwzgrbim5wj44etgq.py
# Source Nodes: [x_13, x_16], Original ATen: [aten.gelu, aten.view]
# x_13 => add_9, erf, mul_10, mul_8, mul_9
# x_16 => view_14
triton_poi_fused_gelu_view_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_youkaichao/ap/capzngy74kdvijnrsbwdxkb6alhtuee2rqioyntbtzk6soav5v64.py
# Source Nodes: [cnn_feat_token], Original ATen: [aten.view]
# cnn_feat_token => view_16
triton_poi_fused_view_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctq37gu74xju7eissdzockymmaznikcbl6apw6fbk7wo3nlj54dt.py
# Source Nodes: [l__mod___blocks_0_1_attn_q, l__mod___blocks_0_1_norm1, x_24], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_1_attn_q => view_19
# l__mod___blocks_0_1_norm1 => add_12, add_13, clone_6, mul_11, mul_12, rsqrt_4, sub_4, var_mean_4
# x_24 => view_22
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (64*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xgcnzacawb2uaysxl3y5rnn5vczbfrav5ffkytidsbzgmc2i4r.py
# Source Nodes: [l__mod___blocks_0_1_norm2, x_31, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_1_norm2 => add_17, add_18, clone_9, mul_15, mul_16, rsqrt_6, sub_6, var_mean_6
# x_31 => add_16
# x_32 => view_30
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwedzb6ejpjytmn3i7hbd6la3na3fakrlnmolrdwf4jhja2dzij5.py
# Source Nodes: [l__mod___blocks_0_2_attn_q, l__mod___blocks_0_2_norm1, x_39, x_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_2_attn_q => view_34
# l__mod___blocks_0_2_norm1 => add_21, add_22, clone_12, mul_20, mul_21, rsqrt_7, sub_7, var_mean_7
# x_39 => add_20
# x_40 => view_37
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tl.store(out_ptr4 + (r1 + (64*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7xmb2k5csrubbhyvhu6dlqqseximac5z4y7wrk5juc6jrp53nx.py
# Source Nodes: [l__mod___blocks_0_2_norm2, x_39, x_47, x_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_2_norm2 => add_26, add_27, clone_15, mul_24, mul_25, rsqrt_9, sub_9, var_mean_9
# x_39 => add_20
# x_47 => add_25
# x_48 => view_45
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3vdy6aoeaqzs5djszzxaq272cul7mivoixwfrhtjn5fezfblbu.py
# Source Nodes: [l__mod___blocks_1_0_attn_q, l__mod___blocks_1_0_norm1, x_59, x_61], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_0_attn_q => view_51
# l__mod___blocks_1_0_norm1 => add_32, add_33, mul_31, mul_32, rsqrt_11, sub_11, var_mean_11
# x_59 => add_30, add_31, clone_18, mul_29, mul_30, rsqrt_10, sub_10, var_mean_10
# x_61 => view_54
triton_red_fused_native_layer_norm_native_layer_norm_backward_view_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp31_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight,
        )
        tmp31_mean = tl.where(rmask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(rmask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(rmask & xmask, tmp31_weight_next, tmp31_weight)
    tmp31_tmp, tmp32_tmp, tmp33_tmp = triton_helpers.welford(
        tmp31_mean, tmp31_m2, tmp31_weight, 1
    )
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    tmp33 = tmp33_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38 - tmp22
        tmp40 = 128.0
        tmp41 = tmp32 / tmp40
        tmp42 = 1e-06
        tmp43 = tmp41 + tmp42
        tmp44 = tl.math.rsqrt(tmp43)
        tmp45 = tmp39 * tmp44
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp45, rmask & xmask)
        tl.store(out_ptr6 + (r2 + (128*x3)), tmp49, rmask & xmask)
        tl.store(out_ptr7 + (r2 + (128*x3)), tmp49, rmask & xmask)
    tmp50 = 128.0
    tmp51 = tmp32 / tmp50
    tmp52 = 1e-06
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp54 / tmp50
    tmp56 = tmp5 / tmp50
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = tl.math.rsqrt(tmp58)
    tmp60 = tmp59 / tmp50
    tl.store(out_ptr8 + (x3), tmp55, xmask)
    tl.store(out_ptr9 + (x3), tmp60, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwo5whf23atqzcdgurqen3uyrtmvfaj46owohadfz4szbkwdsu7t.py
# Source Nodes: [q_3, x_64], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
# q_3 => permute_37
# x_64 => _scaled_dot_product_efficient_attention_3
triton_poi_fused__scaled_dot_product_efficient_attention_permute_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_permute_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2*x2) + (128*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (64*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6464uug4mx2axgxj77yis2zswd3jrjermrgkt2o7d2psnbprcw.py
# Source Nodes: [l__mod___blocks_1_0_attn_kv, x_63], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_0_attn_kv => view_56
# x_63 => add_34, add_35, clone_20, mul_33, mul_34, rsqrt_12, sub_12, var_mean_12
triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp7 = tl.load(in_ptr0 + (x0 + (49*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp20, rmask & xmask)
    tmp21 = 128.0
    tmp22 = tmp5 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr4 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7oylghq4fjzmf5pwcyoxs222uldvgobics3uvcip3rgnu7atytv.py
# Source Nodes: [l__mod___blocks_1_0_norm2, x_59, x_68, x_69], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_0_norm2 => add_37, add_38, mul_35, mul_36, rsqrt_13, sub_13, var_mean_13
# x_59 => add_31, mul_30
# x_68 => add_36
# x_69 => view_62
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkslax4xl4cjsf6vqigkrvvqksfevxhd5co33gf6jxmxq3xprij.py
# Source Nodes: [x_70, x_73], Original ATen: [aten.gelu, aten.view]
# x_70 => add_39, erf_3, mul_37, mul_38, mul_39
# x_73 => view_64
triton_poi_fused_gelu_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lxfitr5e6esszgryh36wkq3jaojjbkhycomibsdws5nfembl3y.py
# Source Nodes: [cnn_feat_token_1], Original ATen: [aten.view]
# cnn_feat_token_1 => view_66
triton_poi_fused_view_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgsxsnj5wu76ymhqyffklvio7474vn32q5r7tmtvniw4ve6lcse.py
# Source Nodes: [l__mod___blocks_1_1_attn_q, l__mod___blocks_1_1_norm1, x_81], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_1_attn_q => view_69
# l__mod___blocks_1_1_norm1 => add_42, add_43, clone_24, mul_40, mul_41, rsqrt_14, sub_14, var_mean_14
# x_81 => view_72
triton_red_fused_native_layer_norm_native_layer_norm_backward_view_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp20, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp24, rmask & xmask)
        tl.store(out_ptr4 + (r2 + (128*x3)), tmp24, rmask & xmask)
    tmp25 = 128.0
    tmp26 = tmp7 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = tl.math.rsqrt(tmp28)
    tmp30 = tmp29 / tmp25
    tl.store(out_ptr5 + (x3), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsfk4xoebylun3rwlqylmwrp55dap42zgvlya2uxqy74kuapell.py
# Source Nodes: [l__mod___blocks_1_1_norm2, x_88, x_89], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_1_norm2 => add_47, add_48, clone_27, mul_44, mul_45, rsqrt_16, sub_16, var_mean_16
# x_88 => add_46
# x_89 => view_80
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/34/c34jc6lp3ywlmlvc2lgodufwbumihjjnizupy5lr27kevqtcaxlq.py
# Source Nodes: [l__mod___blocks_1_2_attn_q, l__mod___blocks_1_2_norm1, x_96, x_97], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_2_attn_q => view_84
# l__mod___blocks_1_2_norm1 => add_51, add_52, clone_30, mul_49, mul_50, rsqrt_17, sub_17, var_mean_17
# x_96 => add_50
# x_97 => view_87
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbggnuaiqzddklmglrbjdvb2qzx3eihh67dlvi7qsvc2dddgmanp.py
# Source Nodes: [l__mod___blocks_1_2_norm2, x_104, x_105, x_96], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_2_norm2 => add_56, add_57, clone_33, mul_53, mul_54, rsqrt_19, sub_19, var_mean_19
# x_104 => add_55
# x_105 => view_95
# x_96 => add_50
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7rkgh6mswvmx4t2up43dcn5lrguf4vczcq5lho2rdi2nuzn3tv.py
# Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
# x_132 => clone_42, var_mean_23
triton_red_fused_native_layer_norm_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_27', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gu/cgu7gi4ud4v2ocykpfofx26vfk2camq6b77cdcrntecmyywde3qi.py
# Source Nodes: [x_132], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_132 => add_69, clone_42, rsqrt_23, var_mean_23
triton_per_fused_native_layer_norm_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_28', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ig/ciggxjx223eqynwztuc3st4t3geceywl6fb6ayxgsqxhyow6ygrw.py
# Source Nodes: [l__mod___blocks_2_0_attn_q, l__mod___blocks_2_0_norm1, x_132, x_134], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_0_attn_q => view_116
# l__mod___blocks_2_0_norm1 => add_71, add_72, mul_69, mul_70, rsqrt_24, sub_24, var_mean_24
# x_132 => add_69, add_70, clone_42, mul_67, mul_68, rsqrt_23, sub_23, var_mean_23
# x_134 => view_119
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 320.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 320, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = tmp31 / tmp6
    tmp34 = 1e-06
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp32 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp36 / tmp6
    tl.store(out_ptr0 + (r2 + (320*x3)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (320*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (320*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr6 + (x3), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2ke2rmfc4qthhuwdxbnlfa24zmvytrtlt675omoetmqn4v5cfn.py
# Source Nodes: [q_7, x_137], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
# q_7 => permute_82
# x_137 => _scaled_dot_product_efficient_attention_7
triton_poi_fused__scaled_dot_product_efficient_attention_permute_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_permute_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7840
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 5
    y1 = (yindex // 5)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (5*x2) + (320*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (64*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgo3esltpqefyexzbqzxfec3kg4vyix7krb72kaji6nbdezjjvr.py
# Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
# x_136 => clone_44, var_mean_25
triton_red_fused_native_layer_norm_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1176
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49) % 3
    x0 = xindex % 49
    x2 = (xindex // 147)
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
        tmp3 = tl.load(in_ptr0 + (x0 + (49*r3) + (5243*x1) + (15680*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3rrvqlcpivtcrm7tvikf4fs65fdjxspvi7udrz3eln56ggfrin.py
# Source Nodes: [x_136], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_136 => add_73, clone_44, rsqrt_25, var_mean_25
triton_per_fused_native_layer_norm_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3e/c3eqni3dtnft3pspdljaq6xrin3o34r2sfzflwmn52jqdi5bg7tb.py
# Source Nodes: [l__mod___blocks_2_0_attn_kv, x_136], Original ATen: [aten.native_layer_norm, aten.view]
# l__mod___blocks_2_0_attn_kv => view_121
# x_136 => add_73, add_74, clone_44, mul_71, mul_72, rsqrt_25, sub_25, var_mean_25
triton_poi_fused_native_layer_norm_view_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 320.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (320*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqwvpvb547ykoiu3h2at3cetwccpj6x7brakkucglk2333yukdt.py
# Source Nodes: [l__mod___blocks_2_0_norm2, x_132, x_141, x_142], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_0_norm2 => add_76, add_77, mul_73, mul_74, rsqrt_26, sub_26, var_mean_26
# x_132 => add_70, mul_68
# x_141 => add_75
# x_142 => view_127
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (320*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmcqr3tshwqk5df4spcb62e3oddo4rbqog5lxbljdq5cl4z34pq.py
# Source Nodes: [x_143, x_146], Original ATen: [aten.gelu, aten.view]
# x_143 => add_78, erf_7, mul_75, mul_76, mul_77
# x_146 => view_129
triton_poi_fused_gelu_view_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
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


# kernel path: /tmp/torchinductor_youkaichao/og/coglt76yt3ziudctrwz5jifiywvlt3zbgugyjzpit3p5gm3jmlbr.py
# Source Nodes: [cnn_feat_token_2], Original ATen: [aten.view]
# cnn_feat_token_2 => view_131
triton_poi_fused_view_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6w7t6tosfj3ja4jisrepprfhemb55nt57yy3yanputwvxs4rbn.py
# Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_2_1_norm1 => clone_48, var_mean_27
triton_red_fused_native_layer_norm_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x4 = (xindex // 3)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r3 + (107*x0) + (320*x4)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp19, xmask)
    tl.store(out_ptr1 + (x5), tmp20, xmask)
    tl.store(out_ptr2 + (x5), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caebyrmdvpfeakwfd2somxjtn2j7fjh4g4jlri6todvhlldggqvi.py
# Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_1_norm1 => add_81, clone_48, rsqrt_27, var_mean_27
triton_per_fused_native_layer_norm_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_38', 'mutated_arg_names': []}
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6czuqxtpxxy5cpw7pmf4vyha6xglf5gbzgjcozmgtzqwkqaxsnn.py
# Source Nodes: [l__mod___blocks_2_1_attn_q, l__mod___blocks_2_1_norm1, x_154], Original ATen: [aten.native_layer_norm, aten.view]
# l__mod___blocks_2_1_attn_q => view_134
# l__mod___blocks_2_1_norm1 => add_81, add_82, clone_48, mul_78, mul_79, rsqrt_27, sub_27, var_mean_27
# x_154 => view_137
triton_poi_fused_native_layer_norm_view_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 320.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (320*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr2 + (x2 + (320*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/copyxppzbxpanqbrb43bktr4724h5khszw2pnxqqztmdomvzuw7n.py
# Source Nodes: [l__mod___blocks_2_1_norm2, x_161, x_162], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_1_norm2 => add_86, add_87, clone_51, mul_82, mul_83, rsqrt_29, sub_29, var_mean_29
# x_161 => add_85
# x_162 => view_145
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (320*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhs5cmovfjshiji2kysc4agr5mowc7inlsyvua3fa6jtgm3pxh2.py
# Source Nodes: [l__mod___blocks_2_2_attn_q, l__mod___blocks_2_2_norm1, x_169, x_170], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_2_attn_q => view_149
# l__mod___blocks_2_2_norm1 => add_90, add_91, clone_54, mul_87, mul_88, rsqrt_30, sub_30, var_mean_30
# x_169 => add_89
# x_170 => view_152
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 1568
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
    tl.store(out_ptr4 + (r1 + (320*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/cea333f3mgfj54zvf74uwywfrhtjcomv5w22mgkhnotvvc37krkt.py
# Source Nodes: [l__mod___blocks_2_2_norm2, x_169, x_177, x_178], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_2_norm2 => add_95, add_96, clone_57, mul_91, mul_92, rsqrt_32, sub_32, var_mean_32
# x_169 => add_89
# x_177 => add_94
# x_178 => view_160
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (320*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg475vqjbx32cxvae2ivszog6ngvkfx5evil67ysr2srxdenkwvb.py
# Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
# x_429 => clone_150, var_mean_78
triton_red_fused_native_layer_norm_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_43', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2jr2hfre4nct6yszxs27eplxbtrninxqng5mgfoz62xti3z7vd.py
# Source Nodes: [x_429], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_429 => add_234, clone_150, rsqrt_78, var_mean_78
triton_per_fused_native_layer_norm_native_layer_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_44', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/se/cse32kht26so37fheq3hx35wglckqltrv255h4kgt4a7dea7cage.py
# Source Nodes: [l__mod___blocks_3_0_attn_q, l__mod___blocks_3_0_norm1, x_429], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_3_0_attn_q => view_391
# l__mod___blocks_3_0_norm1 => add_236, add_237, mul_233, mul_234, rsqrt_79, sub_79, var_mean_79
# x_429 => add_234, add_235, clone_150, mul_231, mul_232, rsqrt_78, sub_78, var_mean_78
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (25088*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = tmp31 / tmp6
    tmp34 = 1e-06
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp32 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp36 / tmp6
    tl.store(out_ptr0 + (r2 + (512*x3)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (512*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf53pjjokkivwtinjewvogpg445lensgbz3hxue4p3757el4uiv3.py
# Source Nodes: [q_25, x_431], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
# q_25 => permute_267
# x_431 => _scaled_dot_product_efficient_attention_25
triton_poi_fused__scaled_dot_product_efficient_attention_permute_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_permute_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (8*x2) + (512*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (64*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/caythclj575houtcfzvkp5xbew2w4tbku2yz2dpnsuaxknt4fz2f.py
# Source Nodes: [l__mod___blocks_3_0_norm2, x_429, x_435, x_436], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_3_0_norm2 => add_239, add_240, mul_235, mul_236, rsqrt_80, sub_80, var_mean_80
# x_429 => add_235, mul_232
# x_435 => add_238
# x_436 => view_400
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
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
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctp3bkmxjhrawlkd26y6rz4fnoi4bquozxuoo5wktb2vncylhx5.py
# Source Nodes: [x_437, x_440], Original ATen: [aten.gelu, aten.view]
# x_437 => add_241, erf_25, mul_237, mul_238, mul_239
# x_440 => view_402
triton_poi_fused_gelu_view_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwnooj6tb7wo25uj446h5rpimo4nttlwvi5yvtxvody7y5pl3v3.py
# Source Nodes: [cnn_feat_token_3], Original ATen: [aten.view]
# cnn_feat_token_3 => view_404
triton_poi_fused_view_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenc4fff2u3btutxro4flt3nqiwxonsvj447qvevsee5szdmkuex.py
# Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_3_1_norm1 => clone_155, var_mean_81
triton_red_fused_native_layer_norm_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    x5 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
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
    tl.store(out_ptr0 + (x5), tmp6, xmask)
    tl.store(out_ptr1 + (x5), tmp7, xmask)
    tl.store(out_ptr2 + (x5), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjicty7ud6qfyymzlo4vuwqn4oirjzw5jenvb46itqgbz56bwm4.py
# Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_3_1_norm1 => add_244, clone_155, rsqrt_81, var_mean_81
triton_per_fused_native_layer_norm_native_layer_norm_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_51', 'mutated_arg_names': []}
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckyrsl3phgaoe73scrdnxsfn6old6bphm5aad2i7smvb5b5fgtrc.py
# Source Nodes: [l__mod___blocks_3_1_attn_q, l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm, aten.view]
# l__mod___blocks_3_1_attn_q => view_407
# l__mod___blocks_3_1_norm1 => add_244, add_245, clone_155, mul_240, mul_241, rsqrt_81, sub_81, var_mean_81
triton_poi_fused_native_layer_norm_view_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxp3husdklxakdxny4cw6p4l22uwcwehoamlauel4oirox72kxfh.py
# Source Nodes: [l__mod___blocks_3_1_norm2, x_452, x_453], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_3_1_norm2 => add_247, add_248, clone_157, mul_242, mul_243, rsqrt_82, sub_82, var_mean_82
# x_452 => add_246
# x_453 => view_416
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (25088*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2hmvmbf5lgqg3m3c7ffoycefuquwhqlwbbftpgy4ckckfgbeg6.py
# Source Nodes: [l__mod___blocks_3_2_attn_q, l__mod___blocks_3_2_norm1, x_460], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_3_2_attn_q => view_420
# l__mod___blocks_3_2_norm1 => add_251, add_252, clone_160, mul_247, mul_248, rsqrt_83, sub_83, var_mean_83
# x_460 => add_250
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
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


# kernel path: /tmp/torchinductor_youkaichao/b5/cb52purymaaqi3cl7s4mstlui5prjyckyy4l4tv6c43p36avf2xs.py
# Source Nodes: [l__mod___blocks_3_2_norm2, x_460, x_465, x_466], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_3_2_norm2 => add_254, add_255, clone_162, mul_249, mul_250, rsqrt_84, sub_84, var_mean_84
# x_460 => add_250
# x_465 => add_253
# x_466 => view_429
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
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
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hdd6gsbda64vou3vab6id7vdbwe4pouaszusexddqyrudp3qwe.py
# Source Nodes: [x_473, x_475], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_473 => add_257
# x_475 => add_258, clone_165, mul_254, rsqrt_85, sub_85, var_mean_85
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 392
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
    tmp28 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3xtpkmbp5yqysyyvkrdwlrcghy5ds6acmnvtt3syoxuavetrjm.py
# Source Nodes: [x_475, x_476], Original ATen: [aten.mean, aten.native_layer_norm]
# x_475 => add_259, mul_255
# x_476 => mean
triton_per_fused_mean_native_layer_norm_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (25088*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzgrva5hhkfpkzt26yb76wxovz3kh7npjzdiaf65otq6g77tvcu.py
# Source Nodes: [], Original ATen: [aten.detach]

triton_poi_fused_detach_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (8*x2) + (512*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckytb2756igpbsq232h4tyqvalm7m2xuqaf7xjsgi5wecnz3g5ty.py
# Source Nodes: [], Original ATen: [aten.detach]

triton_poi_fused_detach_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7840
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 5
    y1 = (yindex // 5)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (5*x2) + (320*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzzs2swlzov6atx3odqef4afzu6kafj5m3h5sspctim4v75cecd.py
# Source Nodes: [], Original ATen: [aten.detach]

triton_poi_fused_detach_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2*x2) + (128*y1)), tmp0, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64), (64, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 64), (64, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (64, 64), (64, 1))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (512, 64), (64, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (64, 512), (512, 1))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, 64), (64, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (128, 64), (64, 1))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (64, 64), (64, 1))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (512, 64), (64, 1))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (64, 512), (512, 1))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, 64), (64, 1))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (128, 64), (64, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (64, 64), (64, 1))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (512, 64), (64, 1))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (64, 512), (512, 1))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128), (128, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (256, 128), (128, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (128, 128), (128, 1))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (1024, 128), (128, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (128, 1024), (1024, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, 128), (128, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (256, 128), (128, 1))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (128, 128), (128, 1))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (1024, 128), (128, 1))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_101, (128, 1024), (1024, 1))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, 128), (128, 1))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (256, 128), (128, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (128, 128), (128, 1))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (1024, 128), (128, 1))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 1024), (1024, 1))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, 128), (128, 1))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (256, 128), (128, 1))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (128, 128), (128, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (1024, 128), (128, 1))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_137, (128, 1024), (1024, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_140, (320, ), (1, ))
    assert_size_stride(primals_141, (320, ), (1, ))
    assert_size_stride(primals_142, (320, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_144, (320, ), (1, ))
    assert_size_stride(primals_145, (320, 320), (320, 1))
    assert_size_stride(primals_146, (320, ), (1, ))
    assert_size_stride(primals_147, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_148, (320, ), (1, ))
    assert_size_stride(primals_149, (320, ), (1, ))
    assert_size_stride(primals_150, (320, ), (1, ))
    assert_size_stride(primals_151, (640, 320), (320, 1))
    assert_size_stride(primals_152, (640, ), (1, ))
    assert_size_stride(primals_153, (320, 320), (320, 1))
    assert_size_stride(primals_154, (320, ), (1, ))
    assert_size_stride(primals_155, (320, ), (1, ))
    assert_size_stride(primals_156, (320, ), (1, ))
    assert_size_stride(primals_157, (1280, 320), (320, 1))
    assert_size_stride(primals_158, (1280, ), (1, ))
    assert_size_stride(primals_159, (320, 1280), (1280, 1))
    assert_size_stride(primals_160, (320, ), (1, ))
    assert_size_stride(primals_161, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (320, ), (1, ))
    assert_size_stride(primals_163, (320, ), (1, ))
    assert_size_stride(primals_164, (320, ), (1, ))
    assert_size_stride(primals_165, (320, 320), (320, 1))
    assert_size_stride(primals_166, (320, ), (1, ))
    assert_size_stride(primals_167, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_168, (320, ), (1, ))
    assert_size_stride(primals_169, (320, ), (1, ))
    assert_size_stride(primals_170, (320, ), (1, ))
    assert_size_stride(primals_171, (640, 320), (320, 1))
    assert_size_stride(primals_172, (640, ), (1, ))
    assert_size_stride(primals_173, (320, 320), (320, 1))
    assert_size_stride(primals_174, (320, ), (1, ))
    assert_size_stride(primals_175, (320, ), (1, ))
    assert_size_stride(primals_176, (320, ), (1, ))
    assert_size_stride(primals_177, (1280, 320), (320, 1))
    assert_size_stride(primals_178, (1280, ), (1, ))
    assert_size_stride(primals_179, (320, 1280), (1280, 1))
    assert_size_stride(primals_180, (320, ), (1, ))
    assert_size_stride(primals_181, (320, ), (1, ))
    assert_size_stride(primals_182, (320, ), (1, ))
    assert_size_stride(primals_183, (320, 320), (320, 1))
    assert_size_stride(primals_184, (320, ), (1, ))
    assert_size_stride(primals_185, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_186, (320, ), (1, ))
    assert_size_stride(primals_187, (320, ), (1, ))
    assert_size_stride(primals_188, (320, ), (1, ))
    assert_size_stride(primals_189, (640, 320), (320, 1))
    assert_size_stride(primals_190, (640, ), (1, ))
    assert_size_stride(primals_191, (320, 320), (320, 1))
    assert_size_stride(primals_192, (320, ), (1, ))
    assert_size_stride(primals_193, (320, ), (1, ))
    assert_size_stride(primals_194, (320, ), (1, ))
    assert_size_stride(primals_195, (1280, 320), (320, 1))
    assert_size_stride(primals_196, (1280, ), (1, ))
    assert_size_stride(primals_197, (320, 1280), (1280, 1))
    assert_size_stride(primals_198, (320, ), (1, ))
    assert_size_stride(primals_199, (320, ), (1, ))
    assert_size_stride(primals_200, (320, ), (1, ))
    assert_size_stride(primals_201, (320, 320), (320, 1))
    assert_size_stride(primals_202, (320, ), (1, ))
    assert_size_stride(primals_203, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_204, (320, ), (1, ))
    assert_size_stride(primals_205, (320, ), (1, ))
    assert_size_stride(primals_206, (320, ), (1, ))
    assert_size_stride(primals_207, (640, 320), (320, 1))
    assert_size_stride(primals_208, (640, ), (1, ))
    assert_size_stride(primals_209, (320, 320), (320, 1))
    assert_size_stride(primals_210, (320, ), (1, ))
    assert_size_stride(primals_211, (320, ), (1, ))
    assert_size_stride(primals_212, (320, ), (1, ))
    assert_size_stride(primals_213, (1280, 320), (320, 1))
    assert_size_stride(primals_214, (1280, ), (1, ))
    assert_size_stride(primals_215, (320, 1280), (1280, 1))
    assert_size_stride(primals_216, (320, ), (1, ))
    assert_size_stride(primals_217, (320, ), (1, ))
    assert_size_stride(primals_218, (320, ), (1, ))
    assert_size_stride(primals_219, (320, 320), (320, 1))
    assert_size_stride(primals_220, (320, ), (1, ))
    assert_size_stride(primals_221, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_222, (320, ), (1, ))
    assert_size_stride(primals_223, (320, ), (1, ))
    assert_size_stride(primals_224, (320, ), (1, ))
    assert_size_stride(primals_225, (640, 320), (320, 1))
    assert_size_stride(primals_226, (640, ), (1, ))
    assert_size_stride(primals_227, (320, 320), (320, 1))
    assert_size_stride(primals_228, (320, ), (1, ))
    assert_size_stride(primals_229, (320, ), (1, ))
    assert_size_stride(primals_230, (320, ), (1, ))
    assert_size_stride(primals_231, (1280, 320), (320, 1))
    assert_size_stride(primals_232, (1280, ), (1, ))
    assert_size_stride(primals_233, (320, 1280), (1280, 1))
    assert_size_stride(primals_234, (320, ), (1, ))
    assert_size_stride(primals_235, (320, ), (1, ))
    assert_size_stride(primals_236, (320, ), (1, ))
    assert_size_stride(primals_237, (320, 320), (320, 1))
    assert_size_stride(primals_238, (320, ), (1, ))
    assert_size_stride(primals_239, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_240, (320, ), (1, ))
    assert_size_stride(primals_241, (320, ), (1, ))
    assert_size_stride(primals_242, (320, ), (1, ))
    assert_size_stride(primals_243, (640, 320), (320, 1))
    assert_size_stride(primals_244, (640, ), (1, ))
    assert_size_stride(primals_245, (320, 320), (320, 1))
    assert_size_stride(primals_246, (320, ), (1, ))
    assert_size_stride(primals_247, (320, ), (1, ))
    assert_size_stride(primals_248, (320, ), (1, ))
    assert_size_stride(primals_249, (1280, 320), (320, 1))
    assert_size_stride(primals_250, (1280, ), (1, ))
    assert_size_stride(primals_251, (320, 1280), (1280, 1))
    assert_size_stride(primals_252, (320, ), (1, ))
    assert_size_stride(primals_253, (320, ), (1, ))
    assert_size_stride(primals_254, (320, ), (1, ))
    assert_size_stride(primals_255, (320, 320), (320, 1))
    assert_size_stride(primals_256, (320, ), (1, ))
    assert_size_stride(primals_257, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_258, (320, ), (1, ))
    assert_size_stride(primals_259, (320, ), (1, ))
    assert_size_stride(primals_260, (320, ), (1, ))
    assert_size_stride(primals_261, (640, 320), (320, 1))
    assert_size_stride(primals_262, (640, ), (1, ))
    assert_size_stride(primals_263, (320, 320), (320, 1))
    assert_size_stride(primals_264, (320, ), (1, ))
    assert_size_stride(primals_265, (320, ), (1, ))
    assert_size_stride(primals_266, (320, ), (1, ))
    assert_size_stride(primals_267, (1280, 320), (320, 1))
    assert_size_stride(primals_268, (1280, ), (1, ))
    assert_size_stride(primals_269, (320, 1280), (1280, 1))
    assert_size_stride(primals_270, (320, ), (1, ))
    assert_size_stride(primals_271, (320, ), (1, ))
    assert_size_stride(primals_272, (320, ), (1, ))
    assert_size_stride(primals_273, (320, 320), (320, 1))
    assert_size_stride(primals_274, (320, ), (1, ))
    assert_size_stride(primals_275, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_276, (320, ), (1, ))
    assert_size_stride(primals_277, (320, ), (1, ))
    assert_size_stride(primals_278, (320, ), (1, ))
    assert_size_stride(primals_279, (640, 320), (320, 1))
    assert_size_stride(primals_280, (640, ), (1, ))
    assert_size_stride(primals_281, (320, 320), (320, 1))
    assert_size_stride(primals_282, (320, ), (1, ))
    assert_size_stride(primals_283, (320, ), (1, ))
    assert_size_stride(primals_284, (320, ), (1, ))
    assert_size_stride(primals_285, (1280, 320), (320, 1))
    assert_size_stride(primals_286, (1280, ), (1, ))
    assert_size_stride(primals_287, (320, 1280), (1280, 1))
    assert_size_stride(primals_288, (320, ), (1, ))
    assert_size_stride(primals_289, (320, ), (1, ))
    assert_size_stride(primals_290, (320, ), (1, ))
    assert_size_stride(primals_291, (320, 320), (320, 1))
    assert_size_stride(primals_292, (320, ), (1, ))
    assert_size_stride(primals_293, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_294, (320, ), (1, ))
    assert_size_stride(primals_295, (320, ), (1, ))
    assert_size_stride(primals_296, (320, ), (1, ))
    assert_size_stride(primals_297, (640, 320), (320, 1))
    assert_size_stride(primals_298, (640, ), (1, ))
    assert_size_stride(primals_299, (320, 320), (320, 1))
    assert_size_stride(primals_300, (320, ), (1, ))
    assert_size_stride(primals_301, (320, ), (1, ))
    assert_size_stride(primals_302, (320, ), (1, ))
    assert_size_stride(primals_303, (1280, 320), (320, 1))
    assert_size_stride(primals_304, (1280, ), (1, ))
    assert_size_stride(primals_305, (320, 1280), (1280, 1))
    assert_size_stride(primals_306, (320, ), (1, ))
    assert_size_stride(primals_307, (320, ), (1, ))
    assert_size_stride(primals_308, (320, ), (1, ))
    assert_size_stride(primals_309, (320, 320), (320, 1))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_311, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_312, (320, ), (1, ))
    assert_size_stride(primals_313, (320, ), (1, ))
    assert_size_stride(primals_314, (320, ), (1, ))
    assert_size_stride(primals_315, (640, 320), (320, 1))
    assert_size_stride(primals_316, (640, ), (1, ))
    assert_size_stride(primals_317, (320, 320), (320, 1))
    assert_size_stride(primals_318, (320, ), (1, ))
    assert_size_stride(primals_319, (320, ), (1, ))
    assert_size_stride(primals_320, (320, ), (1, ))
    assert_size_stride(primals_321, (1280, 320), (320, 1))
    assert_size_stride(primals_322, (1280, ), (1, ))
    assert_size_stride(primals_323, (320, 1280), (1280, 1))
    assert_size_stride(primals_324, (320, ), (1, ))
    assert_size_stride(primals_325, (320, ), (1, ))
    assert_size_stride(primals_326, (320, ), (1, ))
    assert_size_stride(primals_327, (320, 320), (320, 1))
    assert_size_stride(primals_328, (320, ), (1, ))
    assert_size_stride(primals_329, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_330, (320, ), (1, ))
    assert_size_stride(primals_331, (320, ), (1, ))
    assert_size_stride(primals_332, (320, ), (1, ))
    assert_size_stride(primals_333, (640, 320), (320, 1))
    assert_size_stride(primals_334, (640, ), (1, ))
    assert_size_stride(primals_335, (320, 320), (320, 1))
    assert_size_stride(primals_336, (320, ), (1, ))
    assert_size_stride(primals_337, (320, ), (1, ))
    assert_size_stride(primals_338, (320, ), (1, ))
    assert_size_stride(primals_339, (1280, 320), (320, 1))
    assert_size_stride(primals_340, (1280, ), (1, ))
    assert_size_stride(primals_341, (320, 1280), (1280, 1))
    assert_size_stride(primals_342, (320, ), (1, ))
    assert_size_stride(primals_343, (320, ), (1, ))
    assert_size_stride(primals_344, (320, ), (1, ))
    assert_size_stride(primals_345, (320, 320), (320, 1))
    assert_size_stride(primals_346, (320, ), (1, ))
    assert_size_stride(primals_347, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_348, (320, ), (1, ))
    assert_size_stride(primals_349, (320, ), (1, ))
    assert_size_stride(primals_350, (320, ), (1, ))
    assert_size_stride(primals_351, (640, 320), (320, 1))
    assert_size_stride(primals_352, (640, ), (1, ))
    assert_size_stride(primals_353, (320, 320), (320, 1))
    assert_size_stride(primals_354, (320, ), (1, ))
    assert_size_stride(primals_355, (320, ), (1, ))
    assert_size_stride(primals_356, (320, ), (1, ))
    assert_size_stride(primals_357, (1280, 320), (320, 1))
    assert_size_stride(primals_358, (1280, ), (1, ))
    assert_size_stride(primals_359, (320, 1280), (1280, 1))
    assert_size_stride(primals_360, (320, ), (1, ))
    assert_size_stride(primals_361, (320, ), (1, ))
    assert_size_stride(primals_362, (320, ), (1, ))
    assert_size_stride(primals_363, (320, 320), (320, 1))
    assert_size_stride(primals_364, (320, ), (1, ))
    assert_size_stride(primals_365, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_366, (320, ), (1, ))
    assert_size_stride(primals_367, (320, ), (1, ))
    assert_size_stride(primals_368, (320, ), (1, ))
    assert_size_stride(primals_369, (640, 320), (320, 1))
    assert_size_stride(primals_370, (640, ), (1, ))
    assert_size_stride(primals_371, (320, 320), (320, 1))
    assert_size_stride(primals_372, (320, ), (1, ))
    assert_size_stride(primals_373, (320, ), (1, ))
    assert_size_stride(primals_374, (320, ), (1, ))
    assert_size_stride(primals_375, (1280, 320), (320, 1))
    assert_size_stride(primals_376, (1280, ), (1, ))
    assert_size_stride(primals_377, (320, 1280), (1280, 1))
    assert_size_stride(primals_378, (320, ), (1, ))
    assert_size_stride(primals_379, (320, ), (1, ))
    assert_size_stride(primals_380, (320, ), (1, ))
    assert_size_stride(primals_381, (320, 320), (320, 1))
    assert_size_stride(primals_382, (320, ), (1, ))
    assert_size_stride(primals_383, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_384, (320, ), (1, ))
    assert_size_stride(primals_385, (320, ), (1, ))
    assert_size_stride(primals_386, (320, ), (1, ))
    assert_size_stride(primals_387, (640, 320), (320, 1))
    assert_size_stride(primals_388, (640, ), (1, ))
    assert_size_stride(primals_389, (320, 320), (320, 1))
    assert_size_stride(primals_390, (320, ), (1, ))
    assert_size_stride(primals_391, (320, ), (1, ))
    assert_size_stride(primals_392, (320, ), (1, ))
    assert_size_stride(primals_393, (1280, 320), (320, 1))
    assert_size_stride(primals_394, (1280, ), (1, ))
    assert_size_stride(primals_395, (320, 1280), (1280, 1))
    assert_size_stride(primals_396, (320, ), (1, ))
    assert_size_stride(primals_397, (320, ), (1, ))
    assert_size_stride(primals_398, (320, ), (1, ))
    assert_size_stride(primals_399, (320, 320), (320, 1))
    assert_size_stride(primals_400, (320, ), (1, ))
    assert_size_stride(primals_401, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_402, (320, ), (1, ))
    assert_size_stride(primals_403, (320, ), (1, ))
    assert_size_stride(primals_404, (320, ), (1, ))
    assert_size_stride(primals_405, (640, 320), (320, 1))
    assert_size_stride(primals_406, (640, ), (1, ))
    assert_size_stride(primals_407, (320, 320), (320, 1))
    assert_size_stride(primals_408, (320, ), (1, ))
    assert_size_stride(primals_409, (320, ), (1, ))
    assert_size_stride(primals_410, (320, ), (1, ))
    assert_size_stride(primals_411, (1280, 320), (320, 1))
    assert_size_stride(primals_412, (1280, ), (1, ))
    assert_size_stride(primals_413, (320, 1280), (1280, 1))
    assert_size_stride(primals_414, (320, ), (1, ))
    assert_size_stride(primals_415, (320, ), (1, ))
    assert_size_stride(primals_416, (320, ), (1, ))
    assert_size_stride(primals_417, (320, 320), (320, 1))
    assert_size_stride(primals_418, (320, ), (1, ))
    assert_size_stride(primals_419, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_420, (320, ), (1, ))
    assert_size_stride(primals_421, (320, ), (1, ))
    assert_size_stride(primals_422, (320, ), (1, ))
    assert_size_stride(primals_423, (640, 320), (320, 1))
    assert_size_stride(primals_424, (640, ), (1, ))
    assert_size_stride(primals_425, (320, 320), (320, 1))
    assert_size_stride(primals_426, (320, ), (1, ))
    assert_size_stride(primals_427, (320, ), (1, ))
    assert_size_stride(primals_428, (320, ), (1, ))
    assert_size_stride(primals_429, (1280, 320), (320, 1))
    assert_size_stride(primals_430, (1280, ), (1, ))
    assert_size_stride(primals_431, (320, 1280), (1280, 1))
    assert_size_stride(primals_432, (320, ), (1, ))
    assert_size_stride(primals_433, (320, ), (1, ))
    assert_size_stride(primals_434, (320, ), (1, ))
    assert_size_stride(primals_435, (320, 320), (320, 1))
    assert_size_stride(primals_436, (320, ), (1, ))
    assert_size_stride(primals_437, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_438, (320, ), (1, ))
    assert_size_stride(primals_439, (320, ), (1, ))
    assert_size_stride(primals_440, (320, ), (1, ))
    assert_size_stride(primals_441, (640, 320), (320, 1))
    assert_size_stride(primals_442, (640, ), (1, ))
    assert_size_stride(primals_443, (320, 320), (320, 1))
    assert_size_stride(primals_444, (320, ), (1, ))
    assert_size_stride(primals_445, (320, ), (1, ))
    assert_size_stride(primals_446, (320, ), (1, ))
    assert_size_stride(primals_447, (1280, 320), (320, 1))
    assert_size_stride(primals_448, (1280, ), (1, ))
    assert_size_stride(primals_449, (320, 1280), (1280, 1))
    assert_size_stride(primals_450, (320, ), (1, ))
    assert_size_stride(primals_451, (320, ), (1, ))
    assert_size_stride(primals_452, (320, ), (1, ))
    assert_size_stride(primals_453, (320, 320), (320, 1))
    assert_size_stride(primals_454, (320, ), (1, ))
    assert_size_stride(primals_455, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_456, (320, ), (1, ))
    assert_size_stride(primals_457, (320, ), (1, ))
    assert_size_stride(primals_458, (320, ), (1, ))
    assert_size_stride(primals_459, (640, 320), (320, 1))
    assert_size_stride(primals_460, (640, ), (1, ))
    assert_size_stride(primals_461, (320, 320), (320, 1))
    assert_size_stride(primals_462, (320, ), (1, ))
    assert_size_stride(primals_463, (320, ), (1, ))
    assert_size_stride(primals_464, (320, ), (1, ))
    assert_size_stride(primals_465, (1280, 320), (320, 1))
    assert_size_stride(primals_466, (1280, ), (1, ))
    assert_size_stride(primals_467, (320, 1280), (1280, 1))
    assert_size_stride(primals_468, (320, ), (1, ))
    assert_size_stride(primals_469, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_470, (512, ), (1, ))
    assert_size_stride(primals_471, (512, ), (1, ))
    assert_size_stride(primals_472, (512, ), (1, ))
    assert_size_stride(primals_473, (512, ), (1, ))
    assert_size_stride(primals_474, (512, ), (1, ))
    assert_size_stride(primals_475, (512, 512), (512, 1))
    assert_size_stride(primals_476, (512, ), (1, ))
    assert_size_stride(primals_477, (1024, 512), (512, 1))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (512, 512), (512, 1))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (512, ), (1, ))
    assert_size_stride(primals_483, (2048, 512), (512, 1))
    assert_size_stride(primals_484, (2048, ), (1, ))
    assert_size_stride(primals_485, (512, 2048), (2048, 1))
    assert_size_stride(primals_486, (512, ), (1, ))
    assert_size_stride(primals_487, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_488, (512, ), (1, ))
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_490, (512, ), (1, ))
    assert_size_stride(primals_491, (512, 512), (512, 1))
    assert_size_stride(primals_492, (512, ), (1, ))
    assert_size_stride(primals_493, (1024, 512), (512, 1))
    assert_size_stride(primals_494, (1024, ), (1, ))
    assert_size_stride(primals_495, (512, 512), (512, 1))
    assert_size_stride(primals_496, (512, ), (1, ))
    assert_size_stride(primals_497, (512, ), (1, ))
    assert_size_stride(primals_498, (512, ), (1, ))
    assert_size_stride(primals_499, (2048, 512), (512, 1))
    assert_size_stride(primals_500, (2048, ), (1, ))
    assert_size_stride(primals_501, (512, 2048), (2048, 1))
    assert_size_stride(primals_502, (512, ), (1, ))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (512, ), (1, ))
    assert_size_stride(primals_505, (512, 512), (512, 1))
    assert_size_stride(primals_506, (512, ), (1, ))
    assert_size_stride(primals_507, (1024, 512), (512, 1))
    assert_size_stride(primals_508, (1024, ), (1, ))
    assert_size_stride(primals_509, (512, 512), (512, 1))
    assert_size_stride(primals_510, (512, ), (1, ))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_512, (512, ), (1, ))
    assert_size_stride(primals_513, (2048, 512), (512, 1))
    assert_size_stride(primals_514, (2048, ), (1, ))
    assert_size_stride(primals_515, (512, 2048), (2048, 1))
    assert_size_stride(primals_516, (512, ), (1, ))
    assert_size_stride(primals_517, (512, ), (1, ))
    assert_size_stride(primals_518, (512, ), (1, ))
    assert_size_stride(primals_519, (1000, 512), (512, 1))
    assert_size_stride(primals_520, (1000, ), (1, ))
    assert_size_stride(primals_521, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 16, grid=grid(192, 16), stream=stream0)
        del primals_1
        buf1 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_9, buf1, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del primals_9
        buf2 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_29, buf2, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del primals_29
        buf3 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_47, buf3, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del primals_47
        buf4 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_61, buf4, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del primals_61
        buf5 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_69, buf5, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del primals_69
        buf6 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_89, buf6, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del primals_89
        buf7 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_107, buf7, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del primals_107
        buf8 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_125, buf8, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del primals_125
        buf9 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_139, buf9, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del primals_139
        buf10 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_147, buf10, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_147
        buf11 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_167, buf11, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_167
        buf12 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_185, buf12, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_185
        buf13 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_203, buf13, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_203
        buf14 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_221, buf14, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_221
        buf15 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_239, buf15, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_239
        buf16 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_257, buf16, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_257
        buf17 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_275, buf17, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_275
        buf18 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_293, buf18, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_293
        buf19 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_311, buf19, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_311
        buf20 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_329, buf20, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_329
        buf21 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_347, buf21, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_347
        buf22 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_365, buf22, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_365
        buf23 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_383, buf23, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_383
        buf24 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_401, buf24, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_401
        buf25 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_419, buf25, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_419
        buf26 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_437, buf26, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_437
        buf27 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_455, buf27, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del primals_455
        buf28 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_469, buf28, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del primals_469
        buf29 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_521, buf29, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_521
        # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf0, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf34 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf38 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf39 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf1084 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        buf1085 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_q, l__mod___blocks_0_0_norm1, x_2, x_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_8.run(buf30, primals_2, primals_3, primals_4, primals_5, primals_6, buf34, buf38, buf39, buf41, buf1084, buf1085, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_2
        del primals_6
        buf40 = reinterpret_tensor(buf30, (25088, 64), (64, 1), 0); del buf30  # reuse
        # Source Nodes: [l__mod___blocks_0_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf39, reinterpret_tensor(primals_7, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf40)
        del primals_8
        # Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, buf1, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf46 = empty((8, 49, 64), device='cuda', dtype=torch.float32)
        buf47 = empty((392, 64), device='cuda', dtype=torch.float32)
        buf1083 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_kv, x_6], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_9.run(buf42, primals_10, primals_11, primals_12, buf46, buf47, buf1083, 392, 64, grid=grid(392), stream=stream0)
        del primals_10
        del primals_12
        buf48 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf47, reinterpret_tensor(primals_13, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf48)
        del primals_14
        # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf49 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf40, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, True)
        buf50 = buf49[0]
        buf51 = buf49[1]
        buf52 = buf49[2]
        buf53 = buf49[3]
        del buf49
        buf54 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_15, (64, 64), (1, 64), 0), out=buf54)
        buf55 = reinterpret_tensor(buf54, (8, 3136, 64), (200704, 64, 1), 0); del buf54  # reuse
        buf59 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf60 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf1082 = reinterpret_tensor(buf42, (8, 3136, 1), (3136, 1, 1), 0); del buf42  # reuse
        # Source Nodes: [l__mod___blocks_0_0_norm2, x_11, x_12, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf55, buf34, primals_3, primals_4, primals_16, primals_17, primals_18, buf59, buf60, buf1082, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_16
        del primals_18
        del primals_4
        buf61 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_20, buf60, reinterpret_tensor(primals_19, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf61)
        del primals_20
        buf62 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_16], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_11.run(buf61, buf62, 12845056, grid=grid(12845056), stream=stream0)
        buf63 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf62, reinterpret_tensor(primals_21, (512, 64), (1, 512), 0), out=buf63)
        buf64 = reinterpret_tensor(buf63, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf63  # reuse
        # Source Nodes: [cnn_feat_token], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf64, buf55, primals_22, 1605632, grid=grid(1605632), stream=stream0)
        del primals_22
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf65, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf69 = buf55; del buf55  # reuse
        buf70 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf1081 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_1_attn_q, l__mod___blocks_0_1_norm1, x_24], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_13.run(buf65, primals_24, buf64, primals_25, primals_26, buf69, buf70, buf72, buf1081, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_26
        buf71 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_28, buf70, reinterpret_tensor(primals_27, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf71)
        del primals_28
        # Source Nodes: [l__mod___blocks_0_1_attn_sr], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, buf2, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf77 = empty((8, 49, 64), device='cuda', dtype=torch.float32)
        buf78 = empty((392, 64), device='cuda', dtype=torch.float32)
        buf1080 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_1_attn_kv, x_26], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_9.run(buf73, primals_30, primals_31, primals_32, buf77, buf78, buf1080, 392, 64, grid=grid(392), stream=stream0)
        del primals_30
        del primals_32
        buf79 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_34, buf78, reinterpret_tensor(primals_33, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf79)
        del primals_34
        # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf80 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf71, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf79, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf79, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, True)
        buf81 = buf80[0]
        buf82 = buf80[1]
        buf83 = buf80[2]
        buf84 = buf80[3]
        del buf80
        buf85 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_35, (64, 64), (1, 64), 0), out=buf85)
        buf86 = reinterpret_tensor(buf85, (8, 3136, 64), (200704, 64, 1), 0); del buf85  # reuse
        buf90 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf91 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf1079 = reinterpret_tensor(buf73, (8, 3136, 1), (3136, 1, 1), 0); del buf73  # reuse
        # Source Nodes: [l__mod___blocks_0_1_norm2, x_31, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf86, buf65, primals_24, buf64, primals_36, primals_37, primals_38, buf90, buf91, buf1079, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_24
        del primals_36
        del primals_38
        buf92 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_40, buf91, reinterpret_tensor(primals_39, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf92)
        del primals_40
        buf93 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_11.run(buf92, buf93, 12845056, grid=grid(12845056), stream=stream0)
        buf94 = reinterpret_tensor(buf65, (25088, 64), (64, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf93, reinterpret_tensor(primals_41, (512, 64), (1, 512), 0), out=buf94)
        buf98 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf99 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf101 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf1078 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2_attn_q, l__mod___blocks_0_2_norm1, x_39, x_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_15.run(buf86, buf94, primals_42, primals_43, primals_44, buf98, buf99, buf101, buf1078, 25088, 64, grid=grid(25088), stream=stream0)
        del primals_44
        buf100 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_46, buf99, reinterpret_tensor(primals_45, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf100)
        del primals_46
        # Source Nodes: [l__mod___blocks_0_2_attn_sr], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, buf3, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf106 = empty((8, 49, 64), device='cuda', dtype=torch.float32)
        buf107 = empty((392, 64), device='cuda', dtype=torch.float32)
        buf1077 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2_attn_kv, x_42], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_9.run(buf102, primals_48, primals_49, primals_50, buf106, buf107, buf1077, 392, 64, grid=grid(392), stream=stream0)
        del primals_48
        del primals_50
        buf108 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_52, buf107, reinterpret_tensor(primals_51, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf108)
        del primals_52
        # Source Nodes: [x_43], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf109 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf100, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf108, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf108, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, True)
        buf110 = buf109[0]
        buf111 = buf109[1]
        buf112 = buf109[2]
        buf113 = buf109[3]
        del buf109
        buf114 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_53, (64, 64), (1, 64), 0), out=buf114)
        buf115 = reinterpret_tensor(buf114, (8, 3136, 64), (200704, 64, 1), 0); del buf114  # reuse
        buf119 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf120 = empty((25088, 64), device='cuda', dtype=torch.float32)
        buf1076 = reinterpret_tensor(buf102, (8, 3136, 1), (3136, 1, 1), 0); del buf102  # reuse
        # Source Nodes: [l__mod___blocks_0_2_norm2, x_39, x_47, x_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16.run(buf115, buf86, buf94, primals_42, primals_54, primals_55, primals_56, buf119, buf120, buf1076, 25088, 64, grid=grid(25088), stream=stream0)
        del buf86
        del primals_42
        del primals_54
        del primals_56
        buf121 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_58, buf120, reinterpret_tensor(primals_57, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf121)
        del primals_58
        buf122 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49, x_52], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_11.run(buf121, buf122, 12845056, grid=grid(12845056), stream=stream0)
        buf123 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf122, reinterpret_tensor(primals_59, (512, 64), (1, 512), 0), out=buf123)
        buf124 = reinterpret_tensor(buf123, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf123  # reuse
        # Source Nodes: [permute_12], Original ATen: [aten.permute]
        triton_poi_fused_view_12.run(buf124, buf115, primals_60, 1605632, grid=grid(1605632), stream=stream0)
        del buf115
        del primals_60
        # Source Nodes: [l__mod___patch_embeds_1_proj], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf4, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf129 = empty((8, 784, 128), device='cuda', dtype=torch.float32)
        buf133 = empty((8, 784, 128), device='cuda', dtype=torch.float32)
        buf134 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        buf1074 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        buf1075 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_attn_q, l__mod___blocks_1_0_norm1, x_59, x_61], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_17.run(buf125, primals_62, primals_63, primals_64, primals_65, primals_66, buf129, buf133, buf134, buf137, buf1074, buf1075, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_62
        del primals_66
        buf135 = reinterpret_tensor(buf125, (6272, 128), (128, 1), 0); del buf125  # reuse
        # Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_68, buf134, reinterpret_tensor(primals_67, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf135)
        del primals_68
        buf136 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf145 = empty_strided((8, 2, 784, 64), (100352, 64, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_3, x_64], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_18.run(buf135, buf136, buf145, 12544, 64, grid=grid(12544, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf5, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf142 = empty((8, 49, 128), device='cuda', dtype=torch.float32)
        buf143 = empty((392, 128), device='cuda', dtype=torch.float32)
        buf1073 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_attn_kv, x_63], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19.run(buf138, primals_70, primals_71, primals_72, buf142, buf143, buf1073, 392, 128, grid=grid(392), stream=stream0)
        del primals_70
        del primals_72
        buf144 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_74, buf143, reinterpret_tensor(primals_73, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf144)
        del primals_74
        # Source Nodes: [x_64], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf146 = aten._scaled_dot_product_efficient_attention(buf145, reinterpret_tensor(buf144, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf144, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, True)
        buf147 = buf146[0]
        buf148 = buf146[1]
        buf149 = buf146[2]
        buf150 = buf146[3]
        del buf146
        buf151 = reinterpret_tensor(buf145, (6272, 128), (128, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_75, (128, 128), (1, 128), 0), out=buf151)
        buf152 = reinterpret_tensor(buf151, (8, 784, 128), (100352, 128, 1), 0); del buf151  # reuse
        buf156 = reinterpret_tensor(buf135, (8, 784, 128), (100352, 128, 1), 0); del buf135  # reuse
        buf157 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf1071 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_norm2, x_59, x_68, x_69], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf152, buf129, primals_63, primals_64, primals_76, primals_77, primals_78, buf156, buf157, buf1071, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_64
        del primals_76
        del primals_78
        buf158 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_80, buf157, reinterpret_tensor(primals_79, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf158)
        del primals_80
        buf159 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70, x_73], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf158, buf159, 6422528, grid=grid(6422528), stream=stream0)
        buf160 = empty((6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf159, reinterpret_tensor(primals_81, (1024, 128), (1, 1024), 0), out=buf160)
        buf161 = reinterpret_tensor(buf160, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf160  # reuse
        # Source Nodes: [cnn_feat_token_1], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf161, buf152, primals_82, 802816, grid=grid(802816), stream=stream0)
        del primals_82
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf162, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf166 = buf152; del buf152  # reuse
        buf167 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        buf1070 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1_attn_q, l__mod___blocks_1_1_norm1, x_81], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_23.run(buf162, primals_84, buf161, primals_85, primals_86, buf166, buf167, buf170, buf1070, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_86
        buf168 = empty((6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf167, reinterpret_tensor(primals_87, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf168)
        del primals_88
        buf169 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((8, 2, 784, 64), (100352, 64, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_4, x_84], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_18.run(buf168, buf169, buf178, 12544, 64, grid=grid(12544, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_1_1_attn_sr], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, buf6, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf175 = reinterpret_tensor(buf138, (8, 49, 128), (6272, 128, 1), 0); del buf138  # reuse
        buf176 = empty((392, 128), device='cuda', dtype=torch.float32)
        buf1069 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1_attn_kv, x_83], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19.run(buf171, primals_90, primals_91, primals_92, buf175, buf176, buf1069, 392, 128, grid=grid(392), stream=stream0)
        del primals_90
        del primals_92
        buf177 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_94, buf176, reinterpret_tensor(primals_93, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf177)
        del primals_94
        # Source Nodes: [x_84], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf179 = aten._scaled_dot_product_efficient_attention(buf178, reinterpret_tensor(buf177, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf177, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, True)
        buf180 = buf179[0]
        buf181 = buf179[1]
        buf182 = buf179[2]
        buf183 = buf179[3]
        del buf179
        buf184 = reinterpret_tensor(buf178, (6272, 128), (128, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_95, (128, 128), (1, 128), 0), out=buf184)
        buf185 = reinterpret_tensor(buf184, (8, 784, 128), (100352, 128, 1), 0); del buf184  # reuse
        buf189 = reinterpret_tensor(buf168, (8, 784, 128), (100352, 128, 1), 0); del buf168  # reuse
        buf190 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf1067 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1_norm2, x_88, x_89], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf185, buf162, primals_84, buf161, primals_96, primals_97, primals_98, buf189, buf190, buf1067, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_84
        del primals_96
        del primals_98
        buf191 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_100, buf190, reinterpret_tensor(primals_99, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf191)
        del primals_100
        buf192 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90, x_93], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf191, buf192, 6422528, grid=grid(6422528), stream=stream0)
        buf193 = reinterpret_tensor(buf162, (6272, 128), (128, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf192, reinterpret_tensor(primals_101, (1024, 128), (1, 1024), 0), out=buf193)
        buf197 = empty((8, 784, 128), device='cuda', dtype=torch.float32)
        buf198 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf201 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        buf1066 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2_attn_q, l__mod___blocks_1_2_norm1, x_96, x_97], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25.run(buf185, buf193, primals_102, primals_103, primals_104, buf197, buf198, buf201, buf1066, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_104
        buf199 = empty((6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_106, buf198, reinterpret_tensor(primals_105, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf199)
        del primals_106
        buf200 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf209 = empty_strided((8, 2, 784, 64), (100352, 64, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_5, x_100], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_18.run(buf199, buf200, buf209, 12544, 64, grid=grid(12544, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_1_2_attn_sr], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, buf7, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf206 = reinterpret_tensor(buf171, (8, 49, 128), (6272, 128, 1), 0); del buf171  # reuse
        buf207 = empty((392, 128), device='cuda', dtype=torch.float32)
        buf1065 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2_attn_kv, x_99], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19.run(buf202, primals_108, primals_109, primals_110, buf206, buf207, buf1065, 392, 128, grid=grid(392), stream=stream0)
        del primals_108
        del primals_110
        buf208 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_112, buf207, reinterpret_tensor(primals_111, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf208)
        del primals_112
        # Source Nodes: [x_100], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf210 = aten._scaled_dot_product_efficient_attention(buf209, reinterpret_tensor(buf208, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf208, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, True)
        buf211 = buf210[0]
        buf212 = buf210[1]
        buf213 = buf210[2]
        buf214 = buf210[3]
        del buf210
        buf215 = reinterpret_tensor(buf209, (6272, 128), (128, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_113, (128, 128), (1, 128), 0), out=buf215)
        buf216 = reinterpret_tensor(buf215, (8, 784, 128), (100352, 128, 1), 0); del buf215  # reuse
        buf220 = reinterpret_tensor(buf199, (8, 784, 128), (100352, 128, 1), 0); del buf199  # reuse
        buf221 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf1063 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2_norm2, x_104, x_105, x_96], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_26.run(buf216, buf185, buf193, primals_102, primals_114, primals_115, primals_116, buf220, buf221, buf1063, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_102
        del primals_114
        del primals_116
        buf222 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_118, buf221, reinterpret_tensor(primals_117, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf222)
        del primals_118
        buf223 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_109], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf222, buf223, 6422528, grid=grid(6422528), stream=stream0)
        buf224 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf223, reinterpret_tensor(primals_119, (1024, 128), (1, 1024), 0), out=buf224)
        buf228 = buf185; del buf185  # reuse
        buf229 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf232 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        buf1062 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_3_attn_q, l__mod___blocks_1_3_norm1, x_112, x_113], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25.run(buf216, buf224, primals_120, primals_121, primals_122, buf228, buf229, buf232, buf1062, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_122
        buf230 = empty((6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_3_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_124, buf229, reinterpret_tensor(primals_123, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf230)
        del primals_124
        buf231 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((8, 2, 784, 64), (100352, 64, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_6, x_116], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_18.run(buf230, buf231, buf240, 12544, 64, grid=grid(12544, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_1_3_attn_sr], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, buf8, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf237 = reinterpret_tensor(buf202, (8, 49, 128), (6272, 128, 1), 0); del buf202  # reuse
        buf238 = empty((392, 128), device='cuda', dtype=torch.float32)
        buf1061 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_3_attn_kv, x_115], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_19.run(buf233, primals_126, primals_127, primals_128, buf237, buf238, buf1061, 392, 128, grid=grid(392), stream=stream0)
        del buf233
        del primals_126
        del primals_128
        buf239 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_3_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf238, reinterpret_tensor(primals_129, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf239)
        del primals_130
        # Source Nodes: [x_116], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf241 = aten._scaled_dot_product_efficient_attention(buf240, reinterpret_tensor(buf239, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf239, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, True)
        buf242 = buf241[0]
        buf243 = buf241[1]
        buf244 = buf241[2]
        buf245 = buf241[3]
        del buf241
        buf246 = reinterpret_tensor(buf240, (6272, 128), (128, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_131, (128, 128), (1, 128), 0), out=buf246)
        buf247 = reinterpret_tensor(buf246, (8, 784, 128), (100352, 128, 1), 0); del buf246  # reuse
        buf251 = reinterpret_tensor(buf230, (8, 784, 128), (100352, 128, 1), 0); del buf230  # reuse
        buf252 = empty((6272, 128), device='cuda', dtype=torch.float32)
        buf1059 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_3_norm2, x_112, x_120, x_121], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_26.run(buf247, buf216, buf224, primals_120, primals_132, primals_133, primals_134, buf251, buf252, buf1059, 6272, 128, grid=grid(6272), stream=stream0)
        del primals_120
        del primals_132
        del primals_134
        buf253 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf252, reinterpret_tensor(primals_135, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf253)
        del primals_136
        buf254 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122, x_125], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf253, buf254, 6422528, grid=grid(6422528), stream=stream0)
        buf255 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf254, reinterpret_tensor(primals_137, (1024, 128), (1, 1024), 0), out=buf255)
        buf256 = reinterpret_tensor(buf255, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf255  # reuse
        # Source Nodes: [permute_29], Original ATen: [aten.permute]
        triton_poi_fused_view_22.run(buf256, buf247, primals_138, 802816, grid=grid(802816), stream=stream0)
        del primals_138
        # Source Nodes: [l__mod___patch_embeds_2_proj], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, buf9, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 320, 14, 14), (62720, 196, 14, 1))
        buf258 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf259 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf260 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_27.run(buf257, primals_140, buf258, buf259, buf260, 4704, 107, grid=grid(4704), stream=stream0)
        buf261 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf262 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf1058 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_28.run(buf258, buf259, buf260, buf261, buf262, buf1058, 1568, 3, grid=grid(1568), stream=stream0)
        buf264 = empty((8, 196, 320), device='cuda', dtype=torch.float32)
        buf268 = empty((8, 196, 320), device='cuda', dtype=torch.float32)
        buf269 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf272 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1057 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_q, l__mod___blocks_2_0_norm1, x_132, x_134], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29.run(buf257, primals_140, buf261, buf262, primals_141, primals_142, primals_143, primals_144, buf264, buf268, buf269, buf272, buf1057, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_140
        del primals_144
        buf270 = reinterpret_tensor(buf257, (1568, 320), (320, 1), 0); del buf257  # reuse
        # Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_146, buf269, reinterpret_tensor(primals_145, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf270)
        del primals_146
        buf271 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf283 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_7, x_137], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf270, buf271, buf283, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, buf10, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf274 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf273, primals_148, buf274, buf275, buf276, 1176, 107, grid=grid(1176), stream=stream0)
        buf277 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf278 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf1056 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf274, buf275, buf276, buf277, buf278, buf1056, 392, 3, grid=grid(392), stream=stream0)
        buf280 = empty((8, 49, 320), device='cuda', dtype=torch.float32)
        buf281 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_kv, x_136], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf273, primals_148, buf277, buf278, primals_149, primals_150, buf280, buf281, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_148
        del primals_150
        buf282 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf281, reinterpret_tensor(primals_151, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf282)
        del primals_152
        # Source Nodes: [x_137], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf284 = aten._scaled_dot_product_efficient_attention(buf283, reinterpret_tensor(buf282, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf282, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf285 = buf284[0]
        buf286 = buf284[1]
        buf287 = buf284[2]
        buf288 = buf284[3]
        del buf284
        buf289 = reinterpret_tensor(buf283, (1568, 320), (320, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_153, (320, 320), (1, 320), 0), out=buf289)
        buf290 = reinterpret_tensor(buf289, (8, 196, 320), (62720, 320, 1), 0); del buf289  # reuse
        buf294 = reinterpret_tensor(buf270, (8, 196, 320), (62720, 320, 1), 0); del buf270  # reuse
        buf295 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1054 = reinterpret_tensor(buf262, (8, 196, 1), (196, 1, 1), 0); del buf262  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm2, x_132, x_141, x_142], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_34.run(buf290, buf264, primals_141, primals_142, primals_154, primals_155, primals_156, buf294, buf295, buf1054, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_142
        del primals_154
        del primals_156
        buf296 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf295, reinterpret_tensor(primals_157, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf296)
        del primals_158
        buf297 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143, x_146], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf296, buf297, 2007040, grid=grid(2007040), stream=stream0)
        buf298 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_159, (1280, 320), (1, 1280), 0), out=buf298)
        buf299 = reinterpret_tensor(buf298, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf298  # reuse
        # Source Nodes: [cnn_feat_token_2], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf299, buf290, primals_160, 501760, grid=grid(501760), stream=stream0)
        del primals_160
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf300, (8, 320, 14, 14), (62720, 196, 14, 1))
        buf301 = reinterpret_tensor(buf260, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf260  # reuse
        buf302 = reinterpret_tensor(buf259, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf259  # reuse
        buf303 = reinterpret_tensor(buf258, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf258  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_37.run(buf300, primals_162, buf299, buf301, buf302, buf303, 4704, 107, grid=grid(4704), stream=stream0)
        buf304 = buf261; del buf261  # reuse
        buf305 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf1053 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_38.run(buf301, buf302, buf303, buf304, buf305, buf1053, 1568, 3, grid=grid(1568), stream=stream0)
        del buf301
        del buf302
        del buf303
        buf307 = buf290; del buf290  # reuse
        buf308 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf311 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1_attn_q, l__mod___blocks_2_1_norm1, x_154], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_39.run(buf300, primals_162, buf299, buf304, buf305, primals_163, primals_164, buf307, buf308, buf311, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del primals_164
        buf309 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_166, buf308, reinterpret_tensor(primals_165, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf309)
        del primals_166
        buf310 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf322 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_8, x_157], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf309, buf310, buf322, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_1_attn_sr], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, buf11, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf313 = buf276; del buf276  # reuse
        buf314 = buf275; del buf275  # reuse
        buf315 = buf274; del buf274  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf312, primals_168, buf313, buf314, buf315, 1176, 107, grid=grid(1176), stream=stream0)
        buf316 = buf278; del buf278  # reuse
        buf317 = buf277; del buf277  # reuse
        buf1052 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf313, buf314, buf315, buf316, buf317, buf1052, 392, 3, grid=grid(392), stream=stream0)
        buf319 = reinterpret_tensor(buf273, (8, 49, 320), (15680, 320, 1), 0); del buf273  # reuse
        buf320 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1_attn_kv, x_156], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf312, primals_168, buf316, buf317, primals_169, primals_170, buf319, buf320, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_168
        del primals_170
        buf321 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf320, reinterpret_tensor(primals_171, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf321)
        del primals_172
        # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf323 = aten._scaled_dot_product_efficient_attention(buf322, reinterpret_tensor(buf321, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf321, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf324 = buf323[0]
        buf325 = buf323[1]
        buf326 = buf323[2]
        buf327 = buf323[3]
        del buf323
        buf328 = reinterpret_tensor(buf322, (1568, 320), (320, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_173, (320, 320), (1, 320), 0), out=buf328)
        buf329 = reinterpret_tensor(buf328, (8, 196, 320), (62720, 320, 1), 0); del buf328  # reuse
        buf333 = reinterpret_tensor(buf309, (8, 196, 320), (62720, 320, 1), 0); del buf309  # reuse
        buf334 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1050 = reinterpret_tensor(buf305, (8, 196, 1), (196, 1, 1), 0); del buf305  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm2, x_161, x_162], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_40.run(buf329, buf300, primals_162, buf299, primals_174, primals_175, primals_176, buf333, buf334, buf1050, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_162
        del primals_174
        del primals_176
        buf335 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_178, buf334, reinterpret_tensor(primals_177, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf335)
        del primals_178
        buf336 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_166], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf335, buf336, 2007040, grid=grid(2007040), stream=stream0)
        buf337 = reinterpret_tensor(buf300, (1568, 320), (320, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf336, reinterpret_tensor(primals_179, (1280, 320), (1, 1280), 0), out=buf337)
        buf341 = empty((8, 196, 320), device='cuda', dtype=torch.float32)
        buf342 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf345 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1049 = reinterpret_tensor(buf304, (8, 196, 1), (196, 1, 1), 0); del buf304  # reuse
        # Source Nodes: [l__mod___blocks_2_2_attn_q, l__mod___blocks_2_2_norm1, x_169, x_170], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf329, buf337, primals_180, primals_181, primals_182, buf341, buf342, buf345, buf1049, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_182
        buf343 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_184, buf342, reinterpret_tensor(primals_183, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf343)
        del primals_184
        buf344 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_9, x_173], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf343, buf344, buf356, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_2_attn_sr], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, buf12, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf347 = buf315; del buf315  # reuse
        buf348 = buf314; del buf314  # reuse
        buf349 = buf313; del buf313  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf346, primals_186, buf347, buf348, buf349, 1176, 107, grid=grid(1176), stream=stream0)
        buf350 = buf317; del buf317  # reuse
        buf351 = buf316; del buf316  # reuse
        buf1048 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf347, buf348, buf349, buf350, buf351, buf1048, 392, 3, grid=grid(392), stream=stream0)
        buf353 = reinterpret_tensor(buf312, (8, 49, 320), (15680, 320, 1), 0); del buf312  # reuse
        buf354 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2_attn_kv, x_172], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf346, primals_186, buf350, buf351, primals_187, primals_188, buf353, buf354, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_186
        del primals_188
        buf355 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_190, buf354, reinterpret_tensor(primals_189, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf355)
        del primals_190
        # Source Nodes: [x_173], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf357 = aten._scaled_dot_product_efficient_attention(buf356, reinterpret_tensor(buf355, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf355, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf358 = buf357[0]
        buf359 = buf357[1]
        buf360 = buf357[2]
        buf361 = buf357[3]
        del buf357
        buf362 = reinterpret_tensor(buf356, (1568, 320), (320, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_191, (320, 320), (1, 320), 0), out=buf362)
        buf363 = reinterpret_tensor(buf362, (8, 196, 320), (62720, 320, 1), 0); del buf362  # reuse
        buf367 = reinterpret_tensor(buf343, (8, 196, 320), (62720, 320, 1), 0); del buf343  # reuse
        buf368 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1046 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2_norm2, x_169, x_177, x_178], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf363, buf329, buf337, primals_180, primals_192, primals_193, primals_194, buf367, buf368, buf1046, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_180
        del primals_192
        del primals_194
        buf369 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_196, buf368, reinterpret_tensor(primals_195, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf369)
        del primals_196
        buf370 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179, x_182], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf369, buf370, 2007040, grid=grid(2007040), stream=stream0)
        buf371 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf370, reinterpret_tensor(primals_197, (1280, 320), (1, 1280), 0), out=buf371)
        buf375 = buf329; del buf329  # reuse
        buf376 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf379 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1045 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_3_attn_q, l__mod___blocks_2_3_norm1, x_185, x_186], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf363, buf371, primals_198, primals_199, primals_200, buf375, buf376, buf379, buf1045, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_200
        buf377 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_3_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_202, buf376, reinterpret_tensor(primals_201, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf377)
        del primals_202
        buf378 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf390 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_10, x_189], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf377, buf378, buf390, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_3_attn_sr], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, buf13, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf381 = buf349; del buf349  # reuse
        buf382 = buf348; del buf348  # reuse
        buf383 = buf347; del buf347  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf380, primals_204, buf381, buf382, buf383, 1176, 107, grid=grid(1176), stream=stream0)
        buf384 = buf351; del buf351  # reuse
        buf385 = buf350; del buf350  # reuse
        buf1044 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf381, buf382, buf383, buf384, buf385, buf1044, 392, 3, grid=grid(392), stream=stream0)
        buf387 = reinterpret_tensor(buf346, (8, 49, 320), (15680, 320, 1), 0); del buf346  # reuse
        buf388 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_3_attn_kv, x_188], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf380, primals_204, buf384, buf385, primals_205, primals_206, buf387, buf388, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_204
        del primals_206
        buf389 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_3_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_208, buf388, reinterpret_tensor(primals_207, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf389)
        del primals_208
        # Source Nodes: [x_189], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf391 = aten._scaled_dot_product_efficient_attention(buf390, reinterpret_tensor(buf389, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf389, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf392 = buf391[0]
        buf393 = buf391[1]
        buf394 = buf391[2]
        buf395 = buf391[3]
        del buf391
        buf396 = reinterpret_tensor(buf390, (1568, 320), (320, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_209, (320, 320), (1, 320), 0), out=buf396)
        buf397 = reinterpret_tensor(buf396, (8, 196, 320), (62720, 320, 1), 0); del buf396  # reuse
        buf401 = reinterpret_tensor(buf377, (8, 196, 320), (62720, 320, 1), 0); del buf377  # reuse
        buf402 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1042 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_3_norm2, x_185, x_193, x_194], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf397, buf363, buf371, primals_198, primals_210, primals_211, primals_212, buf401, buf402, buf1042, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_198
        del primals_210
        del primals_212
        buf403 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_214, buf402, reinterpret_tensor(primals_213, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf403)
        del primals_214
        buf404 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195, x_198], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf403, buf404, 2007040, grid=grid(2007040), stream=stream0)
        buf405 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf404, reinterpret_tensor(primals_215, (1280, 320), (1, 1280), 0), out=buf405)
        buf409 = buf363; del buf363  # reuse
        buf410 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf413 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1041 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_4_attn_q, l__mod___blocks_2_4_norm1, x_201, x_202], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf397, buf405, primals_216, primals_217, primals_218, buf409, buf410, buf413, buf1041, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_218
        buf411 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_4_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_220, buf410, reinterpret_tensor(primals_219, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf411)
        del primals_220
        buf412 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf424 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_11, x_205], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf411, buf412, buf424, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_4_attn_sr], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, buf14, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf415 = buf383; del buf383  # reuse
        buf416 = buf382; del buf382  # reuse
        buf417 = buf381; del buf381  # reuse
        # Source Nodes: [x_204], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf414, primals_222, buf415, buf416, buf417, 1176, 107, grid=grid(1176), stream=stream0)
        buf418 = buf385; del buf385  # reuse
        buf419 = buf384; del buf384  # reuse
        buf1040 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf415, buf416, buf417, buf418, buf419, buf1040, 392, 3, grid=grid(392), stream=stream0)
        buf421 = reinterpret_tensor(buf380, (8, 49, 320), (15680, 320, 1), 0); del buf380  # reuse
        buf422 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_4_attn_kv, x_204], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf414, primals_222, buf418, buf419, primals_223, primals_224, buf421, buf422, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_222
        del primals_224
        buf423 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_4_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_226, buf422, reinterpret_tensor(primals_225, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf423)
        del primals_226
        # Source Nodes: [x_205], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf425 = aten._scaled_dot_product_efficient_attention(buf424, reinterpret_tensor(buf423, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf423, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf426 = buf425[0]
        buf427 = buf425[1]
        buf428 = buf425[2]
        buf429 = buf425[3]
        del buf425
        buf430 = reinterpret_tensor(buf424, (1568, 320), (320, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf426, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_227, (320, 320), (1, 320), 0), out=buf430)
        buf431 = reinterpret_tensor(buf430, (8, 196, 320), (62720, 320, 1), 0); del buf430  # reuse
        buf435 = reinterpret_tensor(buf411, (8, 196, 320), (62720, 320, 1), 0); del buf411  # reuse
        buf436 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1038 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_4_norm2, x_201, x_209, x_210], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf431, buf397, buf405, primals_216, primals_228, primals_229, primals_230, buf435, buf436, buf1038, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_216
        del primals_228
        del primals_230
        buf437 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_232, buf436, reinterpret_tensor(primals_231, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf437)
        del primals_232
        buf438 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211, x_214], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf437, buf438, 2007040, grid=grid(2007040), stream=stream0)
        buf439 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf438, reinterpret_tensor(primals_233, (1280, 320), (1, 1280), 0), out=buf439)
        buf443 = buf397; del buf397  # reuse
        buf444 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf447 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1037 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_5_attn_q, l__mod___blocks_2_5_norm1, x_217, x_218], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf431, buf439, primals_234, primals_235, primals_236, buf443, buf444, buf447, buf1037, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_236
        buf445 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_5_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_238, buf444, reinterpret_tensor(primals_237, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf445)
        del primals_238
        buf446 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf458 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_12, x_221], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf445, buf446, buf458, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_5_attn_sr], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, buf15, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf449 = buf417; del buf417  # reuse
        buf450 = buf416; del buf416  # reuse
        buf451 = buf415; del buf415  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf448, primals_240, buf449, buf450, buf451, 1176, 107, grid=grid(1176), stream=stream0)
        buf452 = buf419; del buf419  # reuse
        buf453 = buf418; del buf418  # reuse
        buf1036 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf449, buf450, buf451, buf452, buf453, buf1036, 392, 3, grid=grid(392), stream=stream0)
        buf455 = reinterpret_tensor(buf414, (8, 49, 320), (15680, 320, 1), 0); del buf414  # reuse
        buf456 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_5_attn_kv, x_220], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf448, primals_240, buf452, buf453, primals_241, primals_242, buf455, buf456, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_240
        del primals_242
        buf457 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_5_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_244, buf456, reinterpret_tensor(primals_243, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf457)
        del primals_244
        # Source Nodes: [x_221], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf459 = aten._scaled_dot_product_efficient_attention(buf458, reinterpret_tensor(buf457, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf457, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf460 = buf459[0]
        buf461 = buf459[1]
        buf462 = buf459[2]
        buf463 = buf459[3]
        del buf459
        buf464 = reinterpret_tensor(buf458, (1568, 320), (320, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_245, (320, 320), (1, 320), 0), out=buf464)
        buf465 = reinterpret_tensor(buf464, (8, 196, 320), (62720, 320, 1), 0); del buf464  # reuse
        buf469 = reinterpret_tensor(buf445, (8, 196, 320), (62720, 320, 1), 0); del buf445  # reuse
        buf470 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1034 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_5_norm2, x_217, x_225, x_226], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf465, buf431, buf439, primals_234, primals_246, primals_247, primals_248, buf469, buf470, buf1034, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_234
        del primals_246
        del primals_248
        buf471 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_250, buf470, reinterpret_tensor(primals_249, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf471)
        del primals_250
        buf472 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227, x_230], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf471, buf472, 2007040, grid=grid(2007040), stream=stream0)
        buf473 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf472, reinterpret_tensor(primals_251, (1280, 320), (1, 1280), 0), out=buf473)
        buf477 = buf431; del buf431  # reuse
        buf478 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf481 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1033 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_6_attn_q, l__mod___blocks_2_6_norm1, x_233, x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf465, buf473, primals_252, primals_253, primals_254, buf477, buf478, buf481, buf1033, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_254
        buf479 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_6_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, buf478, reinterpret_tensor(primals_255, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf479)
        del primals_256
        buf480 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf492 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_13, x_237], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf479, buf480, buf492, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_6_attn_sr], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, buf16, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf483 = buf451; del buf451  # reuse
        buf484 = buf450; del buf450  # reuse
        buf485 = buf449; del buf449  # reuse
        # Source Nodes: [x_236], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf482, primals_258, buf483, buf484, buf485, 1176, 107, grid=grid(1176), stream=stream0)
        buf486 = buf453; del buf453  # reuse
        buf487 = buf452; del buf452  # reuse
        buf1032 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf483, buf484, buf485, buf486, buf487, buf1032, 392, 3, grid=grid(392), stream=stream0)
        buf489 = reinterpret_tensor(buf448, (8, 49, 320), (15680, 320, 1), 0); del buf448  # reuse
        buf490 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_6_attn_kv, x_236], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf482, primals_258, buf486, buf487, primals_259, primals_260, buf489, buf490, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_258
        del primals_260
        buf491 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_6_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_262, buf490, reinterpret_tensor(primals_261, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf491)
        del primals_262
        # Source Nodes: [x_237], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf493 = aten._scaled_dot_product_efficient_attention(buf492, reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf494 = buf493[0]
        buf495 = buf493[1]
        buf496 = buf493[2]
        buf497 = buf493[3]
        del buf493
        buf498 = reinterpret_tensor(buf492, (1568, 320), (320, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_263, (320, 320), (1, 320), 0), out=buf498)
        buf499 = reinterpret_tensor(buf498, (8, 196, 320), (62720, 320, 1), 0); del buf498  # reuse
        buf503 = reinterpret_tensor(buf479, (8, 196, 320), (62720, 320, 1), 0); del buf479  # reuse
        buf504 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1030 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_6_norm2, x_233, x_241, x_242], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf499, buf465, buf473, primals_252, primals_264, primals_265, primals_266, buf503, buf504, buf1030, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_252
        del primals_264
        del primals_266
        buf505 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_268, buf504, reinterpret_tensor(primals_267, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf505)
        del primals_268
        buf506 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243, x_246], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf505, buf506, 2007040, grid=grid(2007040), stream=stream0)
        buf507 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf506, reinterpret_tensor(primals_269, (1280, 320), (1, 1280), 0), out=buf507)
        buf511 = buf465; del buf465  # reuse
        buf512 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf515 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1029 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_7_attn_q, l__mod___blocks_2_7_norm1, x_249, x_250], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf499, buf507, primals_270, primals_271, primals_272, buf511, buf512, buf515, buf1029, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_272
        buf513 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_7_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_274, buf512, reinterpret_tensor(primals_273, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf513)
        del primals_274
        buf514 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf526 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_14, x_253], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf513, buf514, buf526, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_7_attn_sr], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, buf17, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf517 = buf485; del buf485  # reuse
        buf518 = buf484; del buf484  # reuse
        buf519 = buf483; del buf483  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf516, primals_276, buf517, buf518, buf519, 1176, 107, grid=grid(1176), stream=stream0)
        buf520 = buf487; del buf487  # reuse
        buf521 = buf486; del buf486  # reuse
        buf1028 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf517, buf518, buf519, buf520, buf521, buf1028, 392, 3, grid=grid(392), stream=stream0)
        buf523 = reinterpret_tensor(buf482, (8, 49, 320), (15680, 320, 1), 0); del buf482  # reuse
        buf524 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_7_attn_kv, x_252], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf516, primals_276, buf520, buf521, primals_277, primals_278, buf523, buf524, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_276
        del primals_278
        buf525 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_7_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_280, buf524, reinterpret_tensor(primals_279, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf525)
        del primals_280
        # Source Nodes: [x_253], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf527 = aten._scaled_dot_product_efficient_attention(buf526, reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf528 = buf527[0]
        buf529 = buf527[1]
        buf530 = buf527[2]
        buf531 = buf527[3]
        del buf527
        buf532 = reinterpret_tensor(buf526, (1568, 320), (320, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_281, (320, 320), (1, 320), 0), out=buf532)
        buf533 = reinterpret_tensor(buf532, (8, 196, 320), (62720, 320, 1), 0); del buf532  # reuse
        buf537 = reinterpret_tensor(buf513, (8, 196, 320), (62720, 320, 1), 0); del buf513  # reuse
        buf538 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1026 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_7_norm2, x_249, x_257, x_258], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf533, buf499, buf507, primals_270, primals_282, primals_283, primals_284, buf537, buf538, buf1026, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_270
        del primals_282
        del primals_284
        buf539 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_286, buf538, reinterpret_tensor(primals_285, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf539)
        del primals_286
        buf540 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259, x_262], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf539, buf540, 2007040, grid=grid(2007040), stream=stream0)
        buf541 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf540, reinterpret_tensor(primals_287, (1280, 320), (1, 1280), 0), out=buf541)
        buf545 = buf499; del buf499  # reuse
        buf546 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf549 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1025 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_8_attn_q, l__mod___blocks_2_8_norm1, x_265, x_266], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf533, buf541, primals_288, primals_289, primals_290, buf545, buf546, buf549, buf1025, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_290
        buf547 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_8_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_292, buf546, reinterpret_tensor(primals_291, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf547)
        del primals_292
        buf548 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf560 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_15, x_269], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf547, buf548, buf560, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_8_attn_sr], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, buf18, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf551 = buf519; del buf519  # reuse
        buf552 = buf518; del buf518  # reuse
        buf553 = buf517; del buf517  # reuse
        # Source Nodes: [x_268], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf550, primals_294, buf551, buf552, buf553, 1176, 107, grid=grid(1176), stream=stream0)
        buf554 = buf521; del buf521  # reuse
        buf555 = buf520; del buf520  # reuse
        buf1024 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf551, buf552, buf553, buf554, buf555, buf1024, 392, 3, grid=grid(392), stream=stream0)
        buf557 = reinterpret_tensor(buf516, (8, 49, 320), (15680, 320, 1), 0); del buf516  # reuse
        buf558 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_8_attn_kv, x_268], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf550, primals_294, buf554, buf555, primals_295, primals_296, buf557, buf558, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_294
        del primals_296
        buf559 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_8_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_298, buf558, reinterpret_tensor(primals_297, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf559)
        del primals_298
        # Source Nodes: [x_269], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf561 = aten._scaled_dot_product_efficient_attention(buf560, reinterpret_tensor(buf559, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf559, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf562 = buf561[0]
        buf563 = buf561[1]
        buf564 = buf561[2]
        buf565 = buf561[3]
        del buf561
        buf566 = reinterpret_tensor(buf560, (1568, 320), (320, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_299, (320, 320), (1, 320), 0), out=buf566)
        buf567 = reinterpret_tensor(buf566, (8, 196, 320), (62720, 320, 1), 0); del buf566  # reuse
        buf571 = reinterpret_tensor(buf547, (8, 196, 320), (62720, 320, 1), 0); del buf547  # reuse
        buf572 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1022 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_8_norm2, x_265, x_273, x_274], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf567, buf533, buf541, primals_288, primals_300, primals_301, primals_302, buf571, buf572, buf1022, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_288
        del primals_300
        del primals_302
        buf573 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_304, buf572, reinterpret_tensor(primals_303, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf573)
        del primals_304
        buf574 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275, x_278], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf573, buf574, 2007040, grid=grid(2007040), stream=stream0)
        buf575 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf574, reinterpret_tensor(primals_305, (1280, 320), (1, 1280), 0), out=buf575)
        buf579 = buf533; del buf533  # reuse
        buf580 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf583 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1021 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_9_attn_q, l__mod___blocks_2_9_norm1, x_281, x_282], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf567, buf575, primals_306, primals_307, primals_308, buf579, buf580, buf583, buf1021, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_308
        buf581 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_9_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_310, buf580, reinterpret_tensor(primals_309, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf581)
        del primals_310
        buf582 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf594 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_16, x_285], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf581, buf582, buf594, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_9_attn_sr], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, buf19, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf585 = buf553; del buf553  # reuse
        buf586 = buf552; del buf552  # reuse
        buf587 = buf551; del buf551  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf584, primals_312, buf585, buf586, buf587, 1176, 107, grid=grid(1176), stream=stream0)
        buf588 = buf555; del buf555  # reuse
        buf589 = buf554; del buf554  # reuse
        buf1020 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf585, buf586, buf587, buf588, buf589, buf1020, 392, 3, grid=grid(392), stream=stream0)
        buf591 = reinterpret_tensor(buf550, (8, 49, 320), (15680, 320, 1), 0); del buf550  # reuse
        buf592 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_9_attn_kv, x_284], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf584, primals_312, buf588, buf589, primals_313, primals_314, buf591, buf592, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_312
        del primals_314
        buf593 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_9_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_316, buf592, reinterpret_tensor(primals_315, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf593)
        del primals_316
        # Source Nodes: [x_285], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf595 = aten._scaled_dot_product_efficient_attention(buf594, reinterpret_tensor(buf593, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf593, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf596 = buf595[0]
        buf597 = buf595[1]
        buf598 = buf595[2]
        buf599 = buf595[3]
        del buf595
        buf600 = reinterpret_tensor(buf594, (1568, 320), (320, 1), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_317, (320, 320), (1, 320), 0), out=buf600)
        buf601 = reinterpret_tensor(buf600, (8, 196, 320), (62720, 320, 1), 0); del buf600  # reuse
        buf605 = reinterpret_tensor(buf581, (8, 196, 320), (62720, 320, 1), 0); del buf581  # reuse
        buf606 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1018 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_9_norm2, x_281, x_289, x_290], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf601, buf567, buf575, primals_306, primals_318, primals_319, primals_320, buf605, buf606, buf1018, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_306
        del primals_318
        del primals_320
        buf607 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_322, buf606, reinterpret_tensor(primals_321, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf607)
        del primals_322
        buf608 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291, x_294], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf607, buf608, 2007040, grid=grid(2007040), stream=stream0)
        buf609 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf608, reinterpret_tensor(primals_323, (1280, 320), (1, 1280), 0), out=buf609)
        buf613 = buf567; del buf567  # reuse
        buf614 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf617 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1017 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_10_attn_q, l__mod___blocks_2_10_norm1, x_297, x_298], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf601, buf609, primals_324, primals_325, primals_326, buf613, buf614, buf617, buf1017, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_326
        buf615 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_10_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_328, buf614, reinterpret_tensor(primals_327, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf615)
        del primals_328
        buf616 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf628 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_17, x_301], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf615, buf616, buf628, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_10_attn_sr], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, buf20, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf619 = buf587; del buf587  # reuse
        buf620 = buf586; del buf586  # reuse
        buf621 = buf585; del buf585  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf618, primals_330, buf619, buf620, buf621, 1176, 107, grid=grid(1176), stream=stream0)
        buf622 = buf589; del buf589  # reuse
        buf623 = buf588; del buf588  # reuse
        buf1016 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf619, buf620, buf621, buf622, buf623, buf1016, 392, 3, grid=grid(392), stream=stream0)
        buf625 = reinterpret_tensor(buf584, (8, 49, 320), (15680, 320, 1), 0); del buf584  # reuse
        buf626 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_10_attn_kv, x_300], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf618, primals_330, buf622, buf623, primals_331, primals_332, buf625, buf626, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_330
        del primals_332
        buf627 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_10_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_334, buf626, reinterpret_tensor(primals_333, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf627)
        del primals_334
        # Source Nodes: [x_301], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf629 = aten._scaled_dot_product_efficient_attention(buf628, reinterpret_tensor(buf627, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf627, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf630 = buf629[0]
        buf631 = buf629[1]
        buf632 = buf629[2]
        buf633 = buf629[3]
        del buf629
        buf634 = reinterpret_tensor(buf628, (1568, 320), (320, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf630, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_335, (320, 320), (1, 320), 0), out=buf634)
        buf635 = reinterpret_tensor(buf634, (8, 196, 320), (62720, 320, 1), 0); del buf634  # reuse
        buf639 = reinterpret_tensor(buf615, (8, 196, 320), (62720, 320, 1), 0); del buf615  # reuse
        buf640 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1014 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_10_norm2, x_297, x_305, x_306], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf635, buf601, buf609, primals_324, primals_336, primals_337, primals_338, buf639, buf640, buf1014, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_324
        del primals_336
        del primals_338
        buf641 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_340, buf640, reinterpret_tensor(primals_339, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf641)
        del primals_340
        buf642 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307, x_310], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf641, buf642, 2007040, grid=grid(2007040), stream=stream0)
        buf643 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf642, reinterpret_tensor(primals_341, (1280, 320), (1, 1280), 0), out=buf643)
        buf647 = buf601; del buf601  # reuse
        buf648 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf651 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1013 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_11_attn_q, l__mod___blocks_2_11_norm1, x_313, x_314], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf635, buf643, primals_342, primals_343, primals_344, buf647, buf648, buf651, buf1013, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_344
        buf649 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_11_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_346, buf648, reinterpret_tensor(primals_345, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf649)
        del primals_346
        buf650 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf662 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_18, x_317], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf649, buf650, buf662, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_11_attn_sr], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, buf21, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf653 = buf621; del buf621  # reuse
        buf654 = buf620; del buf620  # reuse
        buf655 = buf619; del buf619  # reuse
        # Source Nodes: [x_316], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf652, primals_348, buf653, buf654, buf655, 1176, 107, grid=grid(1176), stream=stream0)
        buf656 = buf623; del buf623  # reuse
        buf657 = buf622; del buf622  # reuse
        buf1012 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf653, buf654, buf655, buf656, buf657, buf1012, 392, 3, grid=grid(392), stream=stream0)
        buf659 = reinterpret_tensor(buf618, (8, 49, 320), (15680, 320, 1), 0); del buf618  # reuse
        buf660 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_11_attn_kv, x_316], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf652, primals_348, buf656, buf657, primals_349, primals_350, buf659, buf660, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_348
        del primals_350
        buf661 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_11_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_352, buf660, reinterpret_tensor(primals_351, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf661)
        del primals_352
        # Source Nodes: [x_317], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf663 = aten._scaled_dot_product_efficient_attention(buf662, reinterpret_tensor(buf661, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf661, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf664 = buf663[0]
        buf665 = buf663[1]
        buf666 = buf663[2]
        buf667 = buf663[3]
        del buf663
        buf668 = reinterpret_tensor(buf662, (1568, 320), (320, 1), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_353, (320, 320), (1, 320), 0), out=buf668)
        buf669 = reinterpret_tensor(buf668, (8, 196, 320), (62720, 320, 1), 0); del buf668  # reuse
        buf673 = reinterpret_tensor(buf649, (8, 196, 320), (62720, 320, 1), 0); del buf649  # reuse
        buf674 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1010 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_11_norm2, x_313, x_321, x_322], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf669, buf635, buf643, primals_342, primals_354, primals_355, primals_356, buf673, buf674, buf1010, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_342
        del primals_354
        del primals_356
        buf675 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_358, buf674, reinterpret_tensor(primals_357, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf675)
        del primals_358
        buf676 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323, x_326], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf675, buf676, 2007040, grid=grid(2007040), stream=stream0)
        buf677 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf676, reinterpret_tensor(primals_359, (1280, 320), (1, 1280), 0), out=buf677)
        buf681 = buf635; del buf635  # reuse
        buf682 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf685 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1009 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_12_attn_q, l__mod___blocks_2_12_norm1, x_329, x_330], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf669, buf677, primals_360, primals_361, primals_362, buf681, buf682, buf685, buf1009, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_362
        buf683 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_12_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_364, buf682, reinterpret_tensor(primals_363, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf683)
        del primals_364
        buf684 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf696 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_19, x_333], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf683, buf684, buf696, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_12_attn_sr], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, buf22, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf687 = buf655; del buf655  # reuse
        buf688 = buf654; del buf654  # reuse
        buf689 = buf653; del buf653  # reuse
        # Source Nodes: [x_332], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf686, primals_366, buf687, buf688, buf689, 1176, 107, grid=grid(1176), stream=stream0)
        buf690 = buf657; del buf657  # reuse
        buf691 = buf656; del buf656  # reuse
        buf1008 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf687, buf688, buf689, buf690, buf691, buf1008, 392, 3, grid=grid(392), stream=stream0)
        buf693 = reinterpret_tensor(buf652, (8, 49, 320), (15680, 320, 1), 0); del buf652  # reuse
        buf694 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_12_attn_kv, x_332], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf686, primals_366, buf690, buf691, primals_367, primals_368, buf693, buf694, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_366
        del primals_368
        buf695 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_12_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_370, buf694, reinterpret_tensor(primals_369, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf695)
        del primals_370
        # Source Nodes: [x_333], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf697 = aten._scaled_dot_product_efficient_attention(buf696, reinterpret_tensor(buf695, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf695, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf698 = buf697[0]
        buf699 = buf697[1]
        buf700 = buf697[2]
        buf701 = buf697[3]
        del buf697
        buf702 = reinterpret_tensor(buf696, (1568, 320), (320, 1), 0); del buf696  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf698, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_371, (320, 320), (1, 320), 0), out=buf702)
        buf703 = reinterpret_tensor(buf702, (8, 196, 320), (62720, 320, 1), 0); del buf702  # reuse
        buf707 = reinterpret_tensor(buf683, (8, 196, 320), (62720, 320, 1), 0); del buf683  # reuse
        buf708 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1006 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_12_norm2, x_329, x_337, x_338], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf703, buf669, buf677, primals_360, primals_372, primals_373, primals_374, buf707, buf708, buf1006, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_360
        del primals_372
        del primals_374
        buf709 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_376, buf708, reinterpret_tensor(primals_375, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf709)
        del primals_376
        buf710 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339, x_342], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf709, buf710, 2007040, grid=grid(2007040), stream=stream0)
        buf711 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf710, reinterpret_tensor(primals_377, (1280, 320), (1, 1280), 0), out=buf711)
        buf715 = buf669; del buf669  # reuse
        buf716 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf719 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1005 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_13_attn_q, l__mod___blocks_2_13_norm1, x_345, x_346], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf703, buf711, primals_378, primals_379, primals_380, buf715, buf716, buf719, buf1005, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_380
        buf717 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_13_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_382, buf716, reinterpret_tensor(primals_381, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf717)
        del primals_382
        buf718 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf730 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_20, x_349], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf717, buf718, buf730, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_13_attn_sr], Original ATen: [aten.convolution]
        buf720 = extern_kernels.convolution(buf719, buf23, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf720, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf721 = buf689; del buf689  # reuse
        buf722 = buf688; del buf688  # reuse
        buf723 = buf687; del buf687  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf720, primals_384, buf721, buf722, buf723, 1176, 107, grid=grid(1176), stream=stream0)
        buf724 = buf691; del buf691  # reuse
        buf725 = buf690; del buf690  # reuse
        buf1004 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf721, buf722, buf723, buf724, buf725, buf1004, 392, 3, grid=grid(392), stream=stream0)
        buf727 = reinterpret_tensor(buf686, (8, 49, 320), (15680, 320, 1), 0); del buf686  # reuse
        buf728 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_13_attn_kv, x_348], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf720, primals_384, buf724, buf725, primals_385, primals_386, buf727, buf728, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_384
        del primals_386
        buf729 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_13_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_388, buf728, reinterpret_tensor(primals_387, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf729)
        del primals_388
        # Source Nodes: [x_349], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf731 = aten._scaled_dot_product_efficient_attention(buf730, reinterpret_tensor(buf729, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf729, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf732 = buf731[0]
        buf733 = buf731[1]
        buf734 = buf731[2]
        buf735 = buf731[3]
        del buf731
        buf736 = reinterpret_tensor(buf730, (1568, 320), (320, 1), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_389, (320, 320), (1, 320), 0), out=buf736)
        buf737 = reinterpret_tensor(buf736, (8, 196, 320), (62720, 320, 1), 0); del buf736  # reuse
        buf741 = reinterpret_tensor(buf717, (8, 196, 320), (62720, 320, 1), 0); del buf717  # reuse
        buf742 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf1002 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_13_norm2, x_345, x_353, x_354], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf737, buf703, buf711, primals_378, primals_390, primals_391, primals_392, buf741, buf742, buf1002, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_378
        del primals_390
        del primals_392
        buf743 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_394, buf742, reinterpret_tensor(primals_393, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf743)
        del primals_394
        buf744 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355, x_358], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf743, buf744, 2007040, grid=grid(2007040), stream=stream0)
        buf745 = buf711; del buf711  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf744, reinterpret_tensor(primals_395, (1280, 320), (1, 1280), 0), out=buf745)
        buf749 = buf703; del buf703  # reuse
        buf750 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf753 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf1001 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_14_attn_q, l__mod___blocks_2_14_norm1, x_361, x_362], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf737, buf745, primals_396, primals_397, primals_398, buf749, buf750, buf753, buf1001, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_398
        buf751 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_14_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_400, buf750, reinterpret_tensor(primals_399, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf751)
        del primals_400
        buf752 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf764 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_21, x_365], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf751, buf752, buf764, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_14_attn_sr], Original ATen: [aten.convolution]
        buf754 = extern_kernels.convolution(buf753, buf24, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf754, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf755 = buf723; del buf723  # reuse
        buf756 = buf722; del buf722  # reuse
        buf757 = buf721; del buf721  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf754, primals_402, buf755, buf756, buf757, 1176, 107, grid=grid(1176), stream=stream0)
        buf758 = buf725; del buf725  # reuse
        buf759 = buf724; del buf724  # reuse
        buf1000 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf755, buf756, buf757, buf758, buf759, buf1000, 392, 3, grid=grid(392), stream=stream0)
        buf761 = reinterpret_tensor(buf720, (8, 49, 320), (15680, 320, 1), 0); del buf720  # reuse
        buf762 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_14_attn_kv, x_364], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf754, primals_402, buf758, buf759, primals_403, primals_404, buf761, buf762, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_402
        del primals_404
        buf763 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_14_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_406, buf762, reinterpret_tensor(primals_405, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf763)
        del primals_406
        # Source Nodes: [x_365], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf765 = aten._scaled_dot_product_efficient_attention(buf764, reinterpret_tensor(buf763, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf763, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf766 = buf765[0]
        buf767 = buf765[1]
        buf768 = buf765[2]
        buf769 = buf765[3]
        del buf765
        buf770 = reinterpret_tensor(buf764, (1568, 320), (320, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf766, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_407, (320, 320), (1, 320), 0), out=buf770)
        buf771 = reinterpret_tensor(buf770, (8, 196, 320), (62720, 320, 1), 0); del buf770  # reuse
        buf775 = reinterpret_tensor(buf751, (8, 196, 320), (62720, 320, 1), 0); del buf751  # reuse
        buf776 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf998 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_14_norm2, x_361, x_369, x_370], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf771, buf737, buf745, primals_396, primals_408, primals_409, primals_410, buf775, buf776, buf998, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_396
        del primals_408
        del primals_410
        buf777 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_412, buf776, reinterpret_tensor(primals_411, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf777)
        del primals_412
        buf778 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371, x_374], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf777, buf778, 2007040, grid=grid(2007040), stream=stream0)
        buf779 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf778, reinterpret_tensor(primals_413, (1280, 320), (1, 1280), 0), out=buf779)
        buf783 = buf737; del buf737  # reuse
        buf784 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf787 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf997 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_15_attn_q, l__mod___blocks_2_15_norm1, x_377, x_378], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf771, buf779, primals_414, primals_415, primals_416, buf783, buf784, buf787, buf997, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_416
        buf785 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_15_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_418, buf784, reinterpret_tensor(primals_417, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf785)
        del primals_418
        buf786 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf798 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_22, x_381], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf785, buf786, buf798, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_15_attn_sr], Original ATen: [aten.convolution]
        buf788 = extern_kernels.convolution(buf787, buf25, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf788, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf789 = buf757; del buf757  # reuse
        buf790 = buf756; del buf756  # reuse
        buf791 = buf755; del buf755  # reuse
        # Source Nodes: [x_380], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf788, primals_420, buf789, buf790, buf791, 1176, 107, grid=grid(1176), stream=stream0)
        buf792 = buf759; del buf759  # reuse
        buf793 = buf758; del buf758  # reuse
        buf996 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_380], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf789, buf790, buf791, buf792, buf793, buf996, 392, 3, grid=grid(392), stream=stream0)
        buf795 = reinterpret_tensor(buf754, (8, 49, 320), (15680, 320, 1), 0); del buf754  # reuse
        buf796 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_15_attn_kv, x_380], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf788, primals_420, buf792, buf793, primals_421, primals_422, buf795, buf796, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_420
        del primals_422
        buf797 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_15_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_424, buf796, reinterpret_tensor(primals_423, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf797)
        del primals_424
        # Source Nodes: [x_381], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf799 = aten._scaled_dot_product_efficient_attention(buf798, reinterpret_tensor(buf797, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf797, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf800 = buf799[0]
        buf801 = buf799[1]
        buf802 = buf799[2]
        buf803 = buf799[3]
        del buf799
        buf804 = reinterpret_tensor(buf798, (1568, 320), (320, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf800, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_425, (320, 320), (1, 320), 0), out=buf804)
        buf805 = reinterpret_tensor(buf804, (8, 196, 320), (62720, 320, 1), 0); del buf804  # reuse
        buf809 = reinterpret_tensor(buf785, (8, 196, 320), (62720, 320, 1), 0); del buf785  # reuse
        buf810 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf994 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_15_norm2, x_377, x_385, x_386], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf805, buf771, buf779, primals_414, primals_426, primals_427, primals_428, buf809, buf810, buf994, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_414
        del primals_426
        del primals_428
        buf811 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_430, buf810, reinterpret_tensor(primals_429, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf811)
        del primals_430
        buf812 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387, x_390], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf811, buf812, 2007040, grid=grid(2007040), stream=stream0)
        buf813 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf812, reinterpret_tensor(primals_431, (1280, 320), (1, 1280), 0), out=buf813)
        buf817 = buf771; del buf771  # reuse
        buf818 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf821 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf993 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_16_attn_q, l__mod___blocks_2_16_norm1, x_393, x_394], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf805, buf813, primals_432, primals_433, primals_434, buf817, buf818, buf821, buf993, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_434
        buf819 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_16_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_436, buf818, reinterpret_tensor(primals_435, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf819)
        del primals_436
        buf820 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf832 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_23, x_397], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf819, buf820, buf832, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_16_attn_sr], Original ATen: [aten.convolution]
        buf822 = extern_kernels.convolution(buf821, buf26, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf822, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf823 = buf791; del buf791  # reuse
        buf824 = buf790; del buf790  # reuse
        buf825 = buf789; del buf789  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf822, primals_438, buf823, buf824, buf825, 1176, 107, grid=grid(1176), stream=stream0)
        buf826 = buf793; del buf793  # reuse
        buf827 = buf792; del buf792  # reuse
        buf992 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf823, buf824, buf825, buf826, buf827, buf992, 392, 3, grid=grid(392), stream=stream0)
        buf829 = reinterpret_tensor(buf788, (8, 49, 320), (15680, 320, 1), 0); del buf788  # reuse
        buf830 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_16_attn_kv, x_396], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf822, primals_438, buf826, buf827, primals_439, primals_440, buf829, buf830, 392, 320, grid=grid(392, 320), stream=stream0)
        del primals_438
        del primals_440
        buf831 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_16_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_442, buf830, reinterpret_tensor(primals_441, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf831)
        del primals_442
        # Source Nodes: [x_397], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf833 = aten._scaled_dot_product_efficient_attention(buf832, reinterpret_tensor(buf831, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf831, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf834 = buf833[0]
        buf835 = buf833[1]
        buf836 = buf833[2]
        buf837 = buf833[3]
        del buf833
        buf838 = reinterpret_tensor(buf832, (1568, 320), (320, 1), 0); del buf832  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf834, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_443, (320, 320), (1, 320), 0), out=buf838)
        buf839 = reinterpret_tensor(buf838, (8, 196, 320), (62720, 320, 1), 0); del buf838  # reuse
        buf843 = reinterpret_tensor(buf819, (8, 196, 320), (62720, 320, 1), 0); del buf819  # reuse
        buf844 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf990 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_16_norm2, x_393, x_401, x_402], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf839, buf805, buf813, primals_432, primals_444, primals_445, primals_446, buf843, buf844, buf990, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_432
        del primals_444
        del primals_446
        buf845 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_402], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_448, buf844, reinterpret_tensor(primals_447, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf845)
        del primals_448
        buf846 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_403, x_406], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf845, buf846, 2007040, grid=grid(2007040), stream=stream0)
        buf847 = buf813; del buf813  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf846, reinterpret_tensor(primals_449, (1280, 320), (1, 1280), 0), out=buf847)
        buf851 = buf805; del buf805  # reuse
        buf852 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf855 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        buf989 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_17_attn_q, l__mod___blocks_2_17_norm1, x_409, x_410], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_41.run(buf839, buf847, primals_450, primals_451, primals_452, buf851, buf852, buf855, buf989, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_452
        buf853 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_17_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_454, buf852, reinterpret_tensor(primals_453, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf853)
        del primals_454
        buf854 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        buf866 = empty_strided((8, 5, 196, 64), (62720, 64, 320, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_24, x_413], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_30.run(buf853, buf854, buf866, 7840, 64, grid=grid(7840, 64), stream=stream0)
        # Source Nodes: [l__mod___blocks_2_17_attn_sr], Original ATen: [aten.convolution]
        buf856 = extern_kernels.convolution(buf855, buf27, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf856, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf857 = buf825; del buf825  # reuse
        buf858 = buf824; del buf824  # reuse
        buf859 = buf823; del buf823  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_31.run(buf856, primals_456, buf857, buf858, buf859, 1176, 107, grid=grid(1176), stream=stream0)
        buf860 = buf827; del buf827  # reuse
        buf861 = buf826; del buf826  # reuse
        buf988 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_412], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_32.run(buf857, buf858, buf859, buf860, buf861, buf988, 392, 3, grid=grid(392), stream=stream0)
        del buf857
        del buf858
        del buf859
        buf863 = reinterpret_tensor(buf822, (8, 49, 320), (15680, 320, 1), 0); del buf822  # reuse
        buf864 = empty((392, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_17_attn_kv, x_412], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_33.run(buf856, primals_456, buf860, buf861, primals_457, primals_458, buf863, buf864, 392, 320, grid=grid(392, 320), stream=stream0)
        del buf856
        del primals_456
        del primals_458
        buf865 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_17_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_460, buf864, reinterpret_tensor(primals_459, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf865)
        del primals_460
        # Source Nodes: [x_413], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf867 = aten._scaled_dot_product_efficient_attention(buf866, reinterpret_tensor(buf865, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf865, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, True)
        buf868 = buf867[0]
        buf869 = buf867[1]
        buf870 = buf867[2]
        buf871 = buf867[3]
        del buf867
        buf872 = reinterpret_tensor(buf866, (1568, 320), (320, 1), 0); del buf866  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf868, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_461, (320, 320), (1, 320), 0), out=buf872)
        buf873 = reinterpret_tensor(buf872, (8, 196, 320), (62720, 320, 1), 0); del buf872  # reuse
        buf877 = reinterpret_tensor(buf853, (8, 196, 320), (62720, 320, 1), 0); del buf853  # reuse
        buf878 = empty((1568, 320), device='cuda', dtype=torch.float32)
        buf986 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_17_norm2, x_409, x_417, x_418], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_42.run(buf873, buf839, buf847, primals_450, primals_462, primals_463, primals_464, buf877, buf878, buf986, 1568, 320, grid=grid(1568), stream=stream0)
        del primals_450
        del primals_462
        del primals_464
        buf879 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_418], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_466, buf878, reinterpret_tensor(primals_465, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf879)
        del primals_466
        buf880 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_419, x_422], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_35.run(buf879, buf880, 2007040, grid=grid(2007040), stream=stream0)
        buf881 = buf847; del buf847  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf880, reinterpret_tensor(primals_467, (1280, 320), (1, 1280), 0), out=buf881)
        buf882 = reinterpret_tensor(buf881, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf881  # reuse
        # Source Nodes: [permute_102], Original ATen: [aten.permute]
        triton_poi_fused_view_36.run(buf882, buf873, primals_468, 501760, grid=grid(501760), stream=stream0)
        del primals_468
        # Source Nodes: [l__mod___patch_embeds_3_proj], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, buf28, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf884 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        buf885 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        buf886 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_43.run(buf883, primals_470, buf884, buf885, buf886, 1568, 128, grid=grid(1568), stream=stream0)
        buf887 = buf861; del buf861  # reuse
        buf888 = buf860; del buf860  # reuse
        buf985 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_44.run(buf884, buf885, buf886, buf887, buf888, buf985, 392, 4, grid=grid(392), stream=stream0)
        buf890 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf894 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf895 = empty((392, 512), device='cuda', dtype=torch.float32)
        buf984 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_0_attn_q, l__mod___blocks_3_0_norm1, x_429], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45.run(buf883, primals_470, buf887, buf888, primals_471, primals_472, primals_473, primals_474, buf890, buf894, buf895, buf984, 392, 512, grid=grid(392), stream=stream0)
        del primals_470
        del primals_474
        buf896 = reinterpret_tensor(buf883, (392, 512), (512, 1), 0); del buf883  # reuse
        # Source Nodes: [l__mod___blocks_3_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_476, buf895, reinterpret_tensor(primals_475, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf896)
        del primals_476
        buf897 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda', dtype=torch.float32)
        buf899 = empty_strided((8, 8, 49, 64), (25088, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_25, x_431], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_46.run(buf896, buf897, buf899, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf898 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_478, buf895, reinterpret_tensor(primals_477, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf898)
        del primals_478
        # Source Nodes: [x_431], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf900 = aten._scaled_dot_product_efficient_attention(buf899, reinterpret_tensor(buf898, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf898, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, True)
        buf901 = buf900[0]
        buf902 = buf900[1]
        buf903 = buf900[2]
        buf904 = buf900[3]
        del buf900
        buf905 = reinterpret_tensor(buf899, (392, 512), (512, 1), 0); del buf899  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf901, (392, 512), (512, 1), 0), reinterpret_tensor(primals_479, (512, 512), (1, 512), 0), out=buf905)
        buf906 = reinterpret_tensor(buf905, (8, 49, 512), (25088, 512, 1), 0); del buf905  # reuse
        buf910 = reinterpret_tensor(buf896, (8, 49, 512), (25088, 512, 1), 0); del buf896  # reuse
        buf911 = empty((392, 512), device='cuda', dtype=torch.float32)
        buf982 = reinterpret_tensor(buf888, (8, 49, 1), (49, 1, 1), 0); del buf888  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm2, x_429, x_435, x_436], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_47.run(buf906, buf890, primals_471, primals_472, primals_480, primals_481, primals_482, buf910, buf911, buf982, 392, 512, grid=grid(392), stream=stream0)
        del primals_472
        del primals_480
        del primals_482
        buf912 = reinterpret_tensor(buf247, (392, 2048), (2048, 1), 0); del buf247  # reuse
        # Source Nodes: [x_436], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_484, buf911, reinterpret_tensor(primals_483, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf912)
        del primals_484
        buf913 = reinterpret_tensor(buf216, (392, 2048), (2048, 1), 0); del buf216  # reuse
        # Source Nodes: [x_437, x_440], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_48.run(buf912, buf913, 802816, grid=grid(802816), stream=stream0)
        buf914 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf913, reinterpret_tensor(primals_485, (2048, 512), (1, 2048), 0), out=buf914)
        buf915 = reinterpret_tensor(buf914, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf914  # reuse
        # Source Nodes: [cnn_feat_token_3], Original ATen: [aten.view]
        triton_poi_fused_view_49.run(buf915, buf906, primals_486, 200704, grid=grid(200704), stream=stream0)
        del primals_486
        # Source Nodes: [x_444], Original ATen: [aten.convolution]
        buf916 = extern_kernels.convolution(buf915, primals_487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf916, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf917 = reinterpret_tensor(buf886, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf886  # reuse
        buf918 = reinterpret_tensor(buf885, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf885  # reuse
        buf919 = reinterpret_tensor(buf884, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf884  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_50.run(buf916, primals_488, buf915, buf917, buf918, buf919, 1568, 128, grid=grid(1568), stream=stream0)
        buf920 = buf887; del buf887  # reuse
        buf921 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf981 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_51.run(buf917, buf918, buf919, buf920, buf921, buf981, 392, 4, grid=grid(392), stream=stream0)
        del buf917
        del buf918
        del buf919
        buf923 = buf906; del buf906  # reuse
        buf924 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_1_attn_q, l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_52.run(buf916, primals_488, buf915, buf920, buf921, primals_489, primals_490, buf923, buf924, 392, 512, grid=grid(392, 512), stream=stream0)
        del primals_490
        buf925 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_492, buf924, reinterpret_tensor(primals_491, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf925)
        del primals_492
        buf926 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda', dtype=torch.float32)
        buf928 = empty_strided((8, 8, 49, 64), (25088, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_26, x_448], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_46.run(buf925, buf926, buf928, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf927 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_494, buf924, reinterpret_tensor(primals_493, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf927)
        del primals_494
        # Source Nodes: [x_448], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf929 = aten._scaled_dot_product_efficient_attention(buf928, reinterpret_tensor(buf927, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf927, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, True)
        buf930 = buf929[0]
        buf931 = buf929[1]
        buf932 = buf929[2]
        buf933 = buf929[3]
        del buf929
        buf934 = reinterpret_tensor(buf928, (392, 512), (512, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf930, (392, 512), (512, 1), 0), reinterpret_tensor(primals_495, (512, 512), (1, 512), 0), out=buf934)
        buf935 = reinterpret_tensor(buf934, (8, 49, 512), (25088, 512, 1), 0); del buf934  # reuse
        buf939 = reinterpret_tensor(buf925, (8, 49, 512), (25088, 512, 1), 0); del buf925  # reuse
        buf940 = empty((392, 512), device='cuda', dtype=torch.float32)
        buf979 = reinterpret_tensor(buf921, (8, 49, 1), (49, 1, 1), 0); del buf921  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm2, x_452, x_453], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_53.run(buf935, buf916, primals_488, buf915, primals_496, primals_497, primals_498, buf939, buf940, buf979, 392, 512, grid=grid(392), stream=stream0)
        del primals_488
        del primals_496
        del primals_498
        buf941 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_453], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_500, buf940, reinterpret_tensor(primals_499, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf941)
        del primals_500
        buf942 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454, x_457], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_48.run(buf941, buf942, 802816, grid=grid(802816), stream=stream0)
        buf943 = reinterpret_tensor(buf916, (392, 512), (512, 1), 0); del buf916  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf942, reinterpret_tensor(primals_501, (2048, 512), (1, 2048), 0), out=buf943)
        buf947 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf948 = empty((392, 512), device='cuda', dtype=torch.float32)
        buf978 = reinterpret_tensor(buf920, (8, 49, 1), (49, 1, 1), 0); del buf920  # reuse
        # Source Nodes: [l__mod___blocks_3_2_attn_q, l__mod___blocks_3_2_norm1, x_460], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54.run(buf935, buf943, primals_502, primals_503, primals_504, buf947, buf948, buf978, 392, 512, grid=grid(392), stream=stream0)
        del primals_504
        buf949 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_506, buf948, reinterpret_tensor(primals_505, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf949)
        del primals_506
        buf950 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda', dtype=torch.float32)
        buf952 = empty_strided((8, 8, 49, 64), (25088, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_27, x_461], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_46.run(buf949, buf950, buf952, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf951 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_508, buf948, reinterpret_tensor(primals_507, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf951)
        del primals_508
        # Source Nodes: [x_461], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf953 = aten._scaled_dot_product_efficient_attention(buf952, reinterpret_tensor(buf951, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf951, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, True)
        buf954 = buf953[0]
        buf955 = buf953[1]
        buf956 = buf953[2]
        buf957 = buf953[3]
        del buf953
        buf958 = reinterpret_tensor(buf952, (392, 512), (512, 1), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf954, (392, 512), (512, 1), 0), reinterpret_tensor(primals_509, (512, 512), (1, 512), 0), out=buf958)
        buf959 = reinterpret_tensor(buf958, (8, 49, 512), (25088, 512, 1), 0); del buf958  # reuse
        buf963 = reinterpret_tensor(buf949, (8, 49, 512), (25088, 512, 1), 0); del buf949  # reuse
        buf964 = empty((392, 512), device='cuda', dtype=torch.float32)
        buf976 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_2_norm2, x_460, x_465, x_466], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55.run(buf959, buf935, buf943, primals_502, primals_510, primals_511, primals_512, buf963, buf964, buf976, 392, 512, grid=grid(392), stream=stream0)
        del primals_502
        del primals_510
        del primals_512
        buf965 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_466], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_514, buf964, reinterpret_tensor(primals_513, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf965)
        del primals_514
        buf966 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467, x_470], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_48.run(buf965, buf966, 802816, grid=grid(802816), stream=stream0)
        buf967 = buf943; del buf943  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf966, reinterpret_tensor(primals_515, (2048, 512), (1, 2048), 0), out=buf967)
        buf971 = buf935; del buf935  # reuse
        buf975 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_473, x_475], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56.run(buf959, buf967, primals_516, buf971, buf975, 392, 512, grid=grid(392), stream=stream0)
        del primals_516
        buf972 = empty((8, 512), device='cuda', dtype=torch.float32)
        buf973 = buf972; del buf972  # reuse
        # Source Nodes: [x_475, x_476], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_57.run(buf973, buf971, primals_517, primals_518, 4096, 49, grid=grid(4096), stream=stream0)
        del primals_518
        buf974 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_520, buf973, reinterpret_tensor(primals_519, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf974)
        del primals_520
        buf977 = reinterpret_tensor(buf967, (8, 8, 49, 64), (25088, 1, 512, 8), 0); del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_58.run(buf954, buf977, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf980 = reinterpret_tensor(buf959, (8, 8, 49, 64), (25088, 1, 512, 8), 0); del buf959  # reuse
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_58.run(buf930, buf980, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf983 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_58.run(buf901, buf983, 3136, 64, grid=grid(3136, 64), stream=stream0)
        buf987 = reinterpret_tensor(buf873, (8, 5, 196, 64), (62720, 1, 320, 5), 0); del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf868, buf987, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf991 = reinterpret_tensor(buf839, (8, 5, 196, 64), (62720, 1, 320, 5), 0); del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf834, buf991, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf995 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf800, buf995, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf999 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf766, buf999, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1003 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf732, buf1003, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1007 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf698, buf1007, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1011 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf664, buf1011, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1015 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf630, buf1015, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1019 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf596, buf1019, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1023 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf562, buf1023, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1027 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf528, buf1027, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1031 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf494, buf1031, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1035 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf460, buf1035, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1039 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf426, buf1039, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1043 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf392, buf1043, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1047 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf358, buf1047, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1051 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf324, buf1051, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1055 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_59.run(buf285, buf1055, 7840, 64, grid=grid(7840, 64), stream=stream0)
        buf1060 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_60.run(buf242, buf1060, 12544, 64, grid=grid(12544, 64), stream=stream0)
        buf1064 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_60.run(buf211, buf1064, 12544, 64, grid=grid(12544, 64), stream=stream0)
        buf1068 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_60.run(buf180, buf1068, 12544, 64, grid=grid(12544, 64), stream=stream0)
        buf1072 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_60.run(buf147, buf1072, 12544, 64, grid=grid(12544, 64), stream=stream0)
        return (buf974, buf0, primals_3, primals_5, buf1, primals_11, primals_17, primals_23, primals_25, buf2, primals_31, primals_37, primals_43, buf3, primals_49, primals_55, buf4, primals_63, primals_65, buf5, primals_71, primals_77, primals_83, primals_85, buf6, primals_91, primals_97, primals_103, buf7, primals_109, primals_115, primals_121, buf8, primals_127, primals_133, buf9, primals_141, primals_143, buf10, primals_149, primals_155, primals_161, primals_163, buf11, primals_169, primals_175, primals_181, buf12, primals_187, primals_193, primals_199, buf13, primals_205, primals_211, primals_217, buf14, primals_223, primals_229, primals_235, buf15, primals_241, primals_247, primals_253, buf16, primals_259, primals_265, primals_271, buf17, primals_277, primals_283, primals_289, buf18, primals_295, primals_301, primals_307, buf19, primals_313, primals_319, primals_325, buf20, primals_331, primals_337, primals_343, buf21, primals_349, primals_355, primals_361, buf22, primals_367, primals_373, primals_379, buf23, primals_385, primals_391, primals_397, buf24, primals_403, primals_409, primals_415, buf25, primals_421, primals_427, primals_433, buf26, primals_439, primals_445, primals_451, buf27, primals_457, primals_463, buf28, primals_471, primals_473, primals_481, primals_487, primals_489, primals_497, primals_503, primals_511, primals_517, buf29, buf34, buf38, buf39, reinterpret_tensor(buf40, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf41, buf46, buf47, reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf51, buf52, buf53, reinterpret_tensor(buf50, (25088, 64), (64, 1), 0), buf59, buf60, buf61, buf62, buf64, buf69, buf70, reinterpret_tensor(buf71, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf72, buf77, buf78, reinterpret_tensor(buf79, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf79, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf82, buf83, buf84, reinterpret_tensor(buf81, (25088, 64), (64, 1), 0), buf90, buf91, buf92, buf93, buf98, buf99, reinterpret_tensor(buf100, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf101, buf106, buf107, reinterpret_tensor(buf108, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf108, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf111, buf112, buf113, reinterpret_tensor(buf110, (25088, 64), (64, 1), 0), buf119, buf120, buf121, buf122, buf124, buf129, buf133, buf134, buf136, buf137, buf142, buf143, reinterpret_tensor(buf144, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf144, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf148, buf149, buf150, reinterpret_tensor(buf147, (6272, 128), (128, 1), 0), buf156, buf157, buf158, buf159, buf161, buf166, buf167, buf169, buf170, buf175, buf176, reinterpret_tensor(buf177, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf177, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf181, buf182, buf183, reinterpret_tensor(buf180, (6272, 128), (128, 1), 0), buf189, buf190, buf191, buf192, buf197, buf198, buf200, buf201, buf206, buf207, reinterpret_tensor(buf208, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf208, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf212, buf213, buf214, reinterpret_tensor(buf211, (6272, 128), (128, 1), 0), buf220, buf221, buf222, buf223, buf228, buf229, buf231, buf232, buf237, buf238, reinterpret_tensor(buf239, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf239, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf243, buf244, buf245, reinterpret_tensor(buf242, (6272, 128), (128, 1), 0), buf251, buf252, buf253, buf254, buf256, buf264, buf268, buf269, buf271, buf272, buf280, buf281, reinterpret_tensor(buf282, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf282, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf286, buf287, buf288, reinterpret_tensor(buf285, (1568, 320), (320, 1), 0), buf294, buf295, buf296, buf297, buf299, buf307, buf308, buf310, buf311, buf319, buf320, reinterpret_tensor(buf321, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf321, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf325, buf326, buf327, reinterpret_tensor(buf324, (1568, 320), (320, 1), 0), buf333, buf334, buf335, buf336, buf341, buf342, buf344, buf345, buf353, buf354, reinterpret_tensor(buf355, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf355, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf359, buf360, buf361, reinterpret_tensor(buf358, (1568, 320), (320, 1), 0), buf367, buf368, buf369, buf370, buf375, buf376, buf378, buf379, buf387, buf388, reinterpret_tensor(buf389, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf389, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf393, buf394, buf395, reinterpret_tensor(buf392, (1568, 320), (320, 1), 0), buf401, buf402, buf403, buf404, buf409, buf410, buf412, buf413, buf421, buf422, reinterpret_tensor(buf423, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf423, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf427, buf428, buf429, reinterpret_tensor(buf426, (1568, 320), (320, 1), 0), buf435, buf436, buf437, buf438, buf443, buf444, buf446, buf447, buf455, buf456, reinterpret_tensor(buf457, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf457, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf461, buf462, buf463, reinterpret_tensor(buf460, (1568, 320), (320, 1), 0), buf469, buf470, buf471, buf472, buf477, buf478, buf480, buf481, buf489, buf490, reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf495, buf496, buf497, reinterpret_tensor(buf494, (1568, 320), (320, 1), 0), buf503, buf504, buf505, buf506, buf511, buf512, buf514, buf515, buf523, buf524, reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf529, buf530, buf531, reinterpret_tensor(buf528, (1568, 320), (320, 1), 0), buf537, buf538, buf539, buf540, buf545, buf546, buf548, buf549, buf557, buf558, reinterpret_tensor(buf559, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf559, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf563, buf564, buf565, reinterpret_tensor(buf562, (1568, 320), (320, 1), 0), buf571, buf572, buf573, buf574, buf579, buf580, buf582, buf583, buf591, buf592, reinterpret_tensor(buf593, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf593, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf597, buf598, buf599, reinterpret_tensor(buf596, (1568, 320), (320, 1), 0), buf605, buf606, buf607, buf608, buf613, buf614, buf616, buf617, buf625, buf626, reinterpret_tensor(buf627, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf627, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf631, buf632, buf633, reinterpret_tensor(buf630, (1568, 320), (320, 1), 0), buf639, buf640, buf641, buf642, buf647, buf648, buf650, buf651, buf659, buf660, reinterpret_tensor(buf661, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf661, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf665, buf666, buf667, reinterpret_tensor(buf664, (1568, 320), (320, 1), 0), buf673, buf674, buf675, buf676, buf681, buf682, buf684, buf685, buf693, buf694, reinterpret_tensor(buf695, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf695, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf699, buf700, buf701, reinterpret_tensor(buf698, (1568, 320), (320, 1), 0), buf707, buf708, buf709, buf710, buf715, buf716, buf718, buf719, buf727, buf728, reinterpret_tensor(buf729, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf729, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf733, buf734, buf735, reinterpret_tensor(buf732, (1568, 320), (320, 1), 0), buf741, buf742, buf743, buf744, buf749, buf750, buf752, buf753, buf761, buf762, reinterpret_tensor(buf763, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf763, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf767, buf768, buf769, reinterpret_tensor(buf766, (1568, 320), (320, 1), 0), buf775, buf776, buf777, buf778, buf783, buf784, buf786, buf787, buf795, buf796, reinterpret_tensor(buf797, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf797, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf801, buf802, buf803, reinterpret_tensor(buf800, (1568, 320), (320, 1), 0), buf809, buf810, buf811, buf812, buf817, buf818, buf820, buf821, buf829, buf830, reinterpret_tensor(buf831, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf831, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf835, buf836, buf837, reinterpret_tensor(buf834, (1568, 320), (320, 1), 0), buf843, buf844, buf845, buf846, buf851, buf852, buf854, buf855, buf863, buf864, reinterpret_tensor(buf865, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf865, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf869, buf870, buf871, reinterpret_tensor(buf868, (1568, 320), (320, 1), 0), buf877, buf878, buf879, buf880, buf882, buf890, buf894, buf895, buf897, reinterpret_tensor(buf898, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf898, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf902, buf903, buf904, reinterpret_tensor(buf901, (392, 512), (512, 1), 0), buf910, buf911, buf912, buf913, buf915, buf923, buf924, buf926, reinterpret_tensor(buf927, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf927, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf931, buf932, buf933, reinterpret_tensor(buf930, (392, 512), (512, 1), 0), buf939, buf940, buf941, buf942, buf947, buf948, buf950, reinterpret_tensor(buf951, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf951, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf955, buf956, buf957, reinterpret_tensor(buf954, (392, 512), (512, 1), 0), buf963, buf964, buf965, buf966, buf971, buf973, reinterpret_tensor(primals_519, (1000, 512), (512, 1), 0), buf975, reinterpret_tensor(primals_515, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_513, (2048, 512), (512, 1), 0), buf976, reinterpret_tensor(primals_509, (512, 512), (512, 1), 0), buf977, reinterpret_tensor(primals_507, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_505, (512, 512), (512, 1), 0), buf978, reinterpret_tensor(primals_501, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_499, (2048, 512), (512, 1), 0), buf979, reinterpret_tensor(primals_495, (512, 512), (512, 1), 0), buf980, reinterpret_tensor(primals_493, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_491, (512, 512), (512, 1), 0), buf981, reinterpret_tensor(primals_485, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_483, (2048, 512), (512, 1), 0), buf982, reinterpret_tensor(primals_479, (512, 512), (512, 1), 0), buf983, reinterpret_tensor(primals_477, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_475, (512, 512), (512, 1), 0), buf984, buf985, reinterpret_tensor(primals_467, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_465, (1280, 320), (320, 1), 0), buf986, reinterpret_tensor(primals_461, (320, 320), (320, 1), 0), buf987, reinterpret_tensor(primals_459, (640, 320), (320, 1), 0), buf988, reinterpret_tensor(primals_453, (320, 320), (320, 1), 0), buf989, reinterpret_tensor(primals_449, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_447, (1280, 320), (320, 1), 0), buf990, reinterpret_tensor(primals_443, (320, 320), (320, 1), 0), buf991, reinterpret_tensor(primals_441, (640, 320), (320, 1), 0), buf992, reinterpret_tensor(primals_435, (320, 320), (320, 1), 0), buf993, reinterpret_tensor(primals_431, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_429, (1280, 320), (320, 1), 0), buf994, reinterpret_tensor(primals_425, (320, 320), (320, 1), 0), buf995, reinterpret_tensor(primals_423, (640, 320), (320, 1), 0), buf996, reinterpret_tensor(primals_417, (320, 320), (320, 1), 0), buf997, reinterpret_tensor(primals_413, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_411, (1280, 320), (320, 1), 0), buf998, reinterpret_tensor(primals_407, (320, 320), (320, 1), 0), buf999, reinterpret_tensor(primals_405, (640, 320), (320, 1), 0), buf1000, reinterpret_tensor(primals_399, (320, 320), (320, 1), 0), buf1001, reinterpret_tensor(primals_395, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_393, (1280, 320), (320, 1), 0), buf1002, reinterpret_tensor(primals_389, (320, 320), (320, 1), 0), buf1003, reinterpret_tensor(primals_387, (640, 320), (320, 1), 0), buf1004, reinterpret_tensor(primals_381, (320, 320), (320, 1), 0), buf1005, reinterpret_tensor(primals_377, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_375, (1280, 320), (320, 1), 0), buf1006, reinterpret_tensor(primals_371, (320, 320), (320, 1), 0), buf1007, reinterpret_tensor(primals_369, (640, 320), (320, 1), 0), buf1008, reinterpret_tensor(primals_363, (320, 320), (320, 1), 0), buf1009, reinterpret_tensor(primals_359, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_357, (1280, 320), (320, 1), 0), buf1010, reinterpret_tensor(primals_353, (320, 320), (320, 1), 0), buf1011, reinterpret_tensor(primals_351, (640, 320), (320, 1), 0), buf1012, reinterpret_tensor(primals_345, (320, 320), (320, 1), 0), buf1013, reinterpret_tensor(primals_341, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_339, (1280, 320), (320, 1), 0), buf1014, reinterpret_tensor(primals_335, (320, 320), (320, 1), 0), buf1015, reinterpret_tensor(primals_333, (640, 320), (320, 1), 0), buf1016, reinterpret_tensor(primals_327, (320, 320), (320, 1), 0), buf1017, reinterpret_tensor(primals_323, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_321, (1280, 320), (320, 1), 0), buf1018, reinterpret_tensor(primals_317, (320, 320), (320, 1), 0), buf1019, reinterpret_tensor(primals_315, (640, 320), (320, 1), 0), buf1020, reinterpret_tensor(primals_309, (320, 320), (320, 1), 0), buf1021, reinterpret_tensor(primals_305, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_303, (1280, 320), (320, 1), 0), buf1022, reinterpret_tensor(primals_299, (320, 320), (320, 1), 0), buf1023, reinterpret_tensor(primals_297, (640, 320), (320, 1), 0), buf1024, reinterpret_tensor(primals_291, (320, 320), (320, 1), 0), buf1025, reinterpret_tensor(primals_287, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_285, (1280, 320), (320, 1), 0), buf1026, reinterpret_tensor(primals_281, (320, 320), (320, 1), 0), buf1027, reinterpret_tensor(primals_279, (640, 320), (320, 1), 0), buf1028, reinterpret_tensor(primals_273, (320, 320), (320, 1), 0), buf1029, reinterpret_tensor(primals_269, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_267, (1280, 320), (320, 1), 0), buf1030, reinterpret_tensor(primals_263, (320, 320), (320, 1), 0), buf1031, reinterpret_tensor(primals_261, (640, 320), (320, 1), 0), buf1032, reinterpret_tensor(primals_255, (320, 320), (320, 1), 0), buf1033, reinterpret_tensor(primals_251, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_249, (1280, 320), (320, 1), 0), buf1034, reinterpret_tensor(primals_245, (320, 320), (320, 1), 0), buf1035, reinterpret_tensor(primals_243, (640, 320), (320, 1), 0), buf1036, reinterpret_tensor(primals_237, (320, 320), (320, 1), 0), buf1037, reinterpret_tensor(primals_233, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_231, (1280, 320), (320, 1), 0), buf1038, reinterpret_tensor(primals_227, (320, 320), (320, 1), 0), buf1039, reinterpret_tensor(primals_225, (640, 320), (320, 1), 0), buf1040, reinterpret_tensor(primals_219, (320, 320), (320, 1), 0), buf1041, reinterpret_tensor(primals_215, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_213, (1280, 320), (320, 1), 0), buf1042, reinterpret_tensor(primals_209, (320, 320), (320, 1), 0), buf1043, reinterpret_tensor(primals_207, (640, 320), (320, 1), 0), buf1044, reinterpret_tensor(primals_201, (320, 320), (320, 1), 0), buf1045, reinterpret_tensor(primals_197, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_195, (1280, 320), (320, 1), 0), buf1046, reinterpret_tensor(primals_191, (320, 320), (320, 1), 0), buf1047, reinterpret_tensor(primals_189, (640, 320), (320, 1), 0), buf1048, reinterpret_tensor(primals_183, (320, 320), (320, 1), 0), buf1049, reinterpret_tensor(primals_179, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_177, (1280, 320), (320, 1), 0), buf1050, reinterpret_tensor(primals_173, (320, 320), (320, 1), 0), buf1051, reinterpret_tensor(primals_171, (640, 320), (320, 1), 0), buf1052, reinterpret_tensor(primals_165, (320, 320), (320, 1), 0), buf1053, reinterpret_tensor(primals_159, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_157, (1280, 320), (320, 1), 0), buf1054, reinterpret_tensor(primals_153, (320, 320), (320, 1), 0), buf1055, reinterpret_tensor(primals_151, (640, 320), (320, 1), 0), buf1056, reinterpret_tensor(primals_145, (320, 320), (320, 1), 0), buf1057, buf1058, reinterpret_tensor(primals_137, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_135, (1024, 128), (128, 1), 0), buf1059, reinterpret_tensor(primals_131, (128, 128), (128, 1), 0), buf1060, reinterpret_tensor(primals_129, (256, 128), (128, 1), 0), buf1061, reinterpret_tensor(primals_123, (128, 128), (128, 1), 0), buf1062, reinterpret_tensor(primals_119, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_117, (1024, 128), (128, 1), 0), buf1063, reinterpret_tensor(primals_113, (128, 128), (128, 1), 0), buf1064, reinterpret_tensor(primals_111, (256, 128), (128, 1), 0), buf1065, reinterpret_tensor(primals_105, (128, 128), (128, 1), 0), buf1066, reinterpret_tensor(primals_101, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_99, (1024, 128), (128, 1), 0), buf1067, reinterpret_tensor(primals_95, (128, 128), (128, 1), 0), buf1068, reinterpret_tensor(primals_93, (256, 128), (128, 1), 0), buf1069, reinterpret_tensor(primals_87, (128, 128), (128, 1), 0), buf1070, reinterpret_tensor(primals_81, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_79, (1024, 128), (128, 1), 0), buf1071, reinterpret_tensor(primals_75, (128, 128), (128, 1), 0), buf1072, reinterpret_tensor(primals_73, (256, 128), (128, 1), 0), buf1073, reinterpret_tensor(primals_67, (128, 128), (128, 1), 0), buf1074, buf1075, reinterpret_tensor(primals_59, (64, 512), (512, 1), 0), reinterpret_tensor(primals_57, (512, 64), (64, 1), 0), buf1076, reinterpret_tensor(primals_53, (64, 64), (64, 1), 0), buf110, reinterpret_tensor(primals_51, (128, 64), (64, 1), 0), buf1077, reinterpret_tensor(primals_45, (64, 64), (64, 1), 0), buf1078, reinterpret_tensor(primals_41, (64, 512), (512, 1), 0), reinterpret_tensor(primals_39, (512, 64), (64, 1), 0), buf1079, reinterpret_tensor(primals_35, (64, 64), (64, 1), 0), buf81, reinterpret_tensor(primals_33, (128, 64), (64, 1), 0), buf1080, reinterpret_tensor(primals_27, (64, 64), (64, 1), 0), buf1081, reinterpret_tensor(primals_21, (64, 512), (512, 1), 0), reinterpret_tensor(primals_19, (512, 64), (64, 1), 0), buf1082, reinterpret_tensor(primals_15, (64, 64), (64, 1), 0), buf50, reinterpret_tensor(primals_13, (128, 64), (64, 1), 0), buf1083, reinterpret_tensor(primals_7, (64, 64), (64, 1), 0), buf1084, buf1085, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
