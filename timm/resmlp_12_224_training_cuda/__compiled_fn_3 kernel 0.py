
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


# kernel path: /tmp/torchinductor_youkaichao/pn/cpn6kj63bwt7vsr6hogorvyrrxatbuvdgzwlpnebfaxquen7sbkz.py
# Source Nodes: [getattr_l__mod___blocks___0___linear_tokens, x], Original ATen: [aten.convolution, aten.view]
# getattr_l__mod___blocks___0___linear_tokens => view_1
# x => convolution
triton_poi_fused_convolution_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    x4 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4 % 384), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4 % 384), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp2
    tmp8 = tmp3 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp8, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5sfe5jcgbmg74mu23rt27b76fsjwjuaudl4aq2o4f3irnkvce2.py
# Source Nodes: [addcmul_1, mul, x_4, x_5], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
# addcmul_1 => add_2, mul_3, mul_4
# mul => mul_2
# x_4 => add_1
# x_5 => clone, view_3
triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 1568
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((196*y0) + (75264*(x1 // 196)) + (x1 % 196)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + ((196*y0) + (75264*(x1 // 196)) + (x1 % 196)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr0 + (y0 + (384*x1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckugmy7grcmvnifwy2xlx5kj3tilhygy4ktiviswf3ackxv3adfp.py
# Source Nodes: [x_5, x_6, x_9], Original ATen: [aten.add, aten.gelu, aten.view]
# x_5 => add_3
# x_6 => add_4, erf, mul_5, mul_6, mul_7
# x_9 => view_5
triton_poi_fused_add_gelu_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq52lqibgbjuamw3uhsd72ordnohmnkbts27oiyc5rsav2wgro4z.py
# Source Nodes: [getattr_l__mod___blocks___1___linear_tokens, mul, mul_1, x_11, x_4], Original ATen: [aten.add, aten.mul, aten.view]
# getattr_l__mod___blocks___1___linear_tokens => view_7
# mul => mul_2
# mul_1 => mul_8
# x_11 => add_5
# x_4 => add_1
triton_poi_fused_add_mul_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3 % 384), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y3 % 384), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = tmp9 + tmp13
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
    tl.store(out_ptr1 + (x2 + (196*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxzixaugoxgdsgzxsc47ukbuiyxbp7smvg7ncfjiuanjs4ztrs7.py
# Source Nodes: [mul_2, mul_3, x_12, x_19], Original ATen: [aten.add, aten.mul]
# mul_2 => mul_11
# mul_3 => mul_17
# x_12 => add_7
# x_19 => add_11
triton_poi_fused_add_mul_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5yf6s5377bk7hxdbcwgvg4eppsmzgfw3kovof3z3c4y55utrab.py
# Source Nodes: [getattr_l__mod___blocks___2___linear_tokens], Original ATen: [aten.view]
# getattr_l__mod___blocks___2___linear_tokens => view_13
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 % 384), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 % 384), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlflv6te5obvsairy56kw7xdkfjuosvq76dreatujrdryqctyi4.py
# Source Nodes: [getattr_l__mod___blocks___3___linear_tokens, mul_4, mul_5, x_20, x_27], Original ATen: [aten.add, aten.mul, aten.view]
# getattr_l__mod___blocks___3___linear_tokens => view_19
# mul_4 => mul_20
# mul_5 => mul_26
# x_20 => add_13
# x_27 => add_17
triton_poi_fused_add_mul_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_6', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y3 % 384), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3 % 384), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = tmp9 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/coww6dppsc5yi4bu7h3xkv5hwy2vsd7clqha565ehr5jeptu6rnd.py
# Source Nodes: [mul_20, mul_21, x_84, x_91], Original ATen: [aten.add, aten.mul]
# mul_20 => mul_92
# mul_21 => mul_98
# x_84 => add_61
# x_91 => add_65
triton_poi_fused_add_mul_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cihekjxypatjqnvzkdev5ss5mfof73ipbks34q6bwinvd6fwm5ca.py
# Source Nodes: [getattr_l__mod___blocks___11___linear_tokens], Original ATen: [aten.view]
# getattr_l__mod___blocks___11___linear_tokens => view_67
triton_poi_fused_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 % 384), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 % 384), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((384*x1) + (75264*(y0 // 384)) + (y0 % 384)), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbfy6q3ibyq74ud63w2y57cfbukr7hghasytxl3qaldapdb4yvp.py
# Source Nodes: [addcmul_23, mul_22, x_92, x_93], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
# addcmul_23 => add_68, mul_102, mul_103
# mul_22 => mul_101
# x_92 => add_67
# x_93 => clone_33, view_69
triton_poi_fused__unsafe_view_add_addcmul_clone_mul_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addcmul_clone_mul_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + ((196*x1) + (75264*(y0 // 196)) + (y0 % 196)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr0 + (x1 + (384*y0)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwf2vzif7op7cwtbq3zukhrs3exotfegss4rm3qdcqvf3daluc2.py
# Source Nodes: [mul_22, mul_23, x_100, x_102, x_103, x_92], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
# mul_22 => mul_101
# mul_23 => mul_107
# x_100 => add_71
# x_102 => add_72, mul_108, mul_109
# x_103 => mean
# x_92 => add_67
triton_red_fused_add_addcmul_mean_mul_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addcmul_mean_mul_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    x4 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    x1 = (xindex // 384) % 2
    x2 = (xindex // 768)
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r3 + (98*x1) + (196*x0) + (75264*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr6 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp11 = tmp9 * tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tmp3 * tmp12
        tmp14 = tmp0 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3x3uc2ss5vpuswwc4trgktd6mlwuh72mtmd4v4kigxwvwctkbc.py
# Source Nodes: [mul_22, mul_23, x_100, x_102, x_103, x_92], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
# mul_22 => mul_101
# mul_23 => mul_107
# x_100 => add_71
# x_102 => add_72, mul_108, mul_109
# x_103 => mean
# x_92 => add_67
triton_per_fused_add_addcmul_mean_mul_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addcmul_mean_mul_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (768*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151 = args
    args.clear()
    assert_size_stride(primals_1, (384, ), (1, ))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_6, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_9, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_12, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_13, (384, ), (1, ))
    assert_size_stride(primals_14, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_15, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_17, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_18, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_21, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_23, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_24, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_26, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_27, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_30, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_31, (384, ), (1, ))
    assert_size_stride(primals_32, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_33, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_36, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_38, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_39, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_42, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_45, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_48, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_51, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_54, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_57, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_60, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_63, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_66, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_69, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_72, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_73, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_74, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_75, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (196, 196), (196, 1))
    assert_size_stride(primals_78, (196, ), (1, ))
    assert_size_stride(primals_79, (1536, 384), (384, 1))
    assert_size_stride(primals_80, (1536, ), (1, ))
    assert_size_stride(primals_81, (384, 1536), (1536, 1))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (196, 196), (196, 1))
    assert_size_stride(primals_84, (196, ), (1, ))
    assert_size_stride(primals_85, (1536, 384), (384, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (384, 1536), (1536, 1))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (196, 196), (196, 1))
    assert_size_stride(primals_90, (196, ), (1, ))
    assert_size_stride(primals_91, (1536, 384), (384, 1))
    assert_size_stride(primals_92, (1536, ), (1, ))
    assert_size_stride(primals_93, (384, 1536), (1536, 1))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (196, 196), (196, 1))
    assert_size_stride(primals_96, (196, ), (1, ))
    assert_size_stride(primals_97, (1536, 384), (384, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (384, 1536), (1536, 1))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (196, 196), (196, 1))
    assert_size_stride(primals_102, (196, ), (1, ))
    assert_size_stride(primals_103, (1536, 384), (384, 1))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_105, (384, 1536), (1536, 1))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (196, 196), (196, 1))
    assert_size_stride(primals_108, (196, ), (1, ))
    assert_size_stride(primals_109, (1536, 384), (384, 1))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (384, 1536), (1536, 1))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (196, 196), (196, 1))
    assert_size_stride(primals_114, (196, ), (1, ))
    assert_size_stride(primals_115, (1536, 384), (384, 1))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (384, 1536), (1536, 1))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (196, 196), (196, 1))
    assert_size_stride(primals_120, (196, ), (1, ))
    assert_size_stride(primals_121, (1536, 384), (384, 1))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_123, (384, 1536), (1536, 1))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (196, 196), (196, 1))
    assert_size_stride(primals_126, (196, ), (1, ))
    assert_size_stride(primals_127, (1536, 384), (384, 1))
    assert_size_stride(primals_128, (1536, ), (1, ))
    assert_size_stride(primals_129, (384, 1536), (1536, 1))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (196, 196), (196, 1))
    assert_size_stride(primals_132, (196, ), (1, ))
    assert_size_stride(primals_133, (1536, 384), (384, 1))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_135, (384, 1536), (1536, 1))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (196, 196), (196, 1))
    assert_size_stride(primals_138, (196, ), (1, ))
    assert_size_stride(primals_139, (1536, 384), (384, 1))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_141, (384, 1536), (1536, 1))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (196, 196), (196, 1))
    assert_size_stride(primals_144, (196, ), (1, ))
    assert_size_stride(primals_145, (1536, 384), (384, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (384, 1536), (1536, 1))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (1000, 384), (384, 1))
    assert_size_stride(primals_150, (1000, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_151, primals_75, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___linear_tokens, x], Original ATen: [aten.convolution, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_view_0.run(buf1, primals_76, primals_2, primals_3, buf2, 602112, grid=grid(602112), stream=stream0)
        del primals_2
        del primals_76
        buf3 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_78, buf2, reinterpret_tensor(primals_77, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf3)
        del primals_78
        buf4 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_1, mul, x_4, x_5], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_5, primals_6, buf1, primals_1, buf3, buf4, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_5
        buf5 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_79, (384, 1536), (1, 384), 0), out=buf5)
        buf6 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_6, x_9], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf5, primals_80, buf6, 2408448, grid=grid(2408448), stream=stream0)
        buf7 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_82, buf6, reinterpret_tensor(primals_81, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf7)
        del primals_82
        buf8 = empty_strided((8, 196, 384), (75264, 1, 196), device='cuda', dtype=torch.float32)
        buf9 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___linear_tokens, mul, mul_1, x_11, x_4], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf1, primals_1, buf3, primals_4, buf7, primals_8, primals_9, buf8, buf9, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_8
        buf10 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf9, reinterpret_tensor(primals_83, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf10)
        del primals_84
        buf11 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_3, mul_2, x_12, x_13], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_11, primals_12, buf8, primals_7, buf10, buf11, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_11
        buf12 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_85, (384, 1536), (1, 384), 0), out=buf12)
        buf13 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_14, x_17], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf12, primals_86, buf13, 2408448, grid=grid(2408448), stream=stream0)
        buf14 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf13, reinterpret_tensor(primals_87, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf14)
        del primals_88
        buf15 = buf8; del buf8  # reuse
        # Source Nodes: [mul_2, mul_3, x_12, x_19], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf15, primals_7, buf10, primals_10, buf14, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf16 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(primals_14, primals_15, buf15, buf16, 602112, grid=grid(602112), stream=stream0)
        del primals_14
        buf17 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_90, buf16, reinterpret_tensor(primals_89, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf17)
        del primals_90
        buf18 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_5, mul_4, x_20, x_21], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_17, primals_18, buf15, primals_13, buf17, buf18, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_17
        buf19 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_91, (384, 1536), (1, 384), 0), out=buf19)
        buf20 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_22, x_25], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf19, primals_92, buf20, 2408448, grid=grid(2408448), stream=stream0)
        buf21 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_94, buf20, reinterpret_tensor(primals_93, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf21)
        del primals_94
        buf22 = buf15; del buf15  # reuse
        buf23 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___linear_tokens, mul_4, mul_5, x_20, x_27], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_6.run(buf22, primals_13, buf17, primals_16, buf21, primals_20, primals_21, buf23, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_20
        buf24 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf23, reinterpret_tensor(primals_95, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf24)
        del primals_96
        buf25 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_7, mul_6, x_28, x_29], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_23, primals_24, buf22, primals_19, buf24, buf25, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_23
        buf26 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_97, (384, 1536), (1, 384), 0), out=buf26)
        buf27 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_30, x_33], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf26, primals_98, buf27, 2408448, grid=grid(2408448), stream=stream0)
        buf28 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_100, buf27, reinterpret_tensor(primals_99, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf28)
        del primals_100
        buf29 = buf22; del buf22  # reuse
        # Source Nodes: [mul_6, mul_7, x_28, x_35], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf29, primals_19, buf24, primals_22, buf28, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf30 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(primals_26, primals_27, buf29, buf30, 602112, grid=grid(602112), stream=stream0)
        del primals_26
        buf31 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf30, reinterpret_tensor(primals_101, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf31)
        del primals_102
        buf32 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_9, mul_8, x_36, x_37], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_29, primals_30, buf29, primals_25, buf31, buf32, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_29
        buf33 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, reinterpret_tensor(primals_103, (384, 1536), (1, 384), 0), out=buf33)
        buf34 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37, x_38, x_41], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf33, primals_104, buf34, 2408448, grid=grid(2408448), stream=stream0)
        buf35 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_106, buf34, reinterpret_tensor(primals_105, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf35)
        del primals_106
        buf36 = buf29; del buf29  # reuse
        buf37 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___linear_tokens, mul_8, mul_9, x_36, x_43], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_6.run(buf36, primals_25, buf31, primals_28, buf35, primals_32, primals_33, buf37, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_32
        buf38 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf37, reinterpret_tensor(primals_107, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf38)
        del primals_108
        buf39 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_11, mul_10, x_44, x_45], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_35, primals_36, buf36, primals_31, buf38, buf39, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_35
        buf40 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, reinterpret_tensor(primals_109, (384, 1536), (1, 384), 0), out=buf40)
        buf41 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_46, x_49], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf40, primals_110, buf41, 2408448, grid=grid(2408448), stream=stream0)
        buf42 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_112, buf41, reinterpret_tensor(primals_111, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf42)
        del primals_112
        buf43 = buf36; del buf36  # reuse
        # Source Nodes: [mul_10, mul_11, x_44, x_51], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf43, primals_31, buf38, primals_34, buf42, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf44 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(primals_38, primals_39, buf43, buf44, 602112, grid=grid(602112), stream=stream0)
        del primals_38
        buf45 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_114, buf44, reinterpret_tensor(primals_113, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf45)
        del primals_114
        buf46 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_13, mul_12, x_52, x_53], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_41, primals_42, buf43, primals_37, buf45, buf46, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_41
        buf47 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_115, (384, 1536), (1, 384), 0), out=buf47)
        buf48 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53, x_54, x_57], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf47, primals_116, buf48, 2408448, grid=grid(2408448), stream=stream0)
        buf49 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_118, buf48, reinterpret_tensor(primals_117, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf49)
        del primals_118
        buf50 = buf43; del buf43  # reuse
        buf51 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___linear_tokens, mul_12, mul_13, x_52, x_59], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_6.run(buf50, primals_37, buf45, primals_40, buf49, primals_44, primals_45, buf51, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_44
        buf52 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf51, reinterpret_tensor(primals_119, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf52)
        del primals_120
        buf53 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_15, mul_14, x_60, x_61], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_47, primals_48, buf50, primals_43, buf52, buf53, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_47
        buf54 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten.mm]
        extern_kernels.mm(buf53, reinterpret_tensor(primals_121, (384, 1536), (1, 384), 0), out=buf54)
        buf55 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_62, x_65], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf54, primals_122, buf55, 2408448, grid=grid(2408448), stream=stream0)
        buf56 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_124, buf55, reinterpret_tensor(primals_123, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf56)
        del primals_124
        buf57 = buf50; del buf50  # reuse
        # Source Nodes: [mul_14, mul_15, x_60, x_67], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf57, primals_43, buf52, primals_46, buf56, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf58 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(primals_50, primals_51, buf57, buf58, 602112, grid=grid(602112), stream=stream0)
        del primals_50
        buf59 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf58, reinterpret_tensor(primals_125, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf59)
        del primals_126
        buf60 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_17, mul_16, x_68, x_69], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_53, primals_54, buf57, primals_49, buf59, buf60, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_53
        buf61 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.mm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_127, (384, 1536), (1, 384), 0), out=buf61)
        buf62 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69, x_70, x_73], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf61, primals_128, buf62, 2408448, grid=grid(2408448), stream=stream0)
        buf63 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf62, reinterpret_tensor(primals_129, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf63)
        del primals_130
        buf64 = buf57; del buf57  # reuse
        buf65 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___linear_tokens, mul_16, mul_17, x_68, x_75], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_6.run(buf64, primals_49, buf59, primals_52, buf63, primals_56, primals_57, buf65, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_56
        buf66 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf65, reinterpret_tensor(primals_131, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf66)
        del primals_132
        buf67 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_19, mul_18, x_76, x_77], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_59, primals_60, buf64, primals_55, buf66, buf67, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_59
        buf68 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.mm]
        extern_kernels.mm(buf67, reinterpret_tensor(primals_133, (384, 1536), (1, 384), 0), out=buf68)
        buf69 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77, x_78, x_81], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf68, primals_134, buf69, 2408448, grid=grid(2408448), stream=stream0)
        buf70 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf69, reinterpret_tensor(primals_135, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf70)
        del primals_136
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [mul_18, mul_19, x_76, x_83], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf71, primals_55, buf66, primals_58, buf70, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf72 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(primals_62, primals_63, buf71, buf72, 602112, grid=grid(602112), stream=stream0)
        del primals_62
        buf73 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_138, buf72, reinterpret_tensor(primals_137, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf73)
        del primals_138
        buf74 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_21, mul_20, x_84, x_85], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_1.run(primals_65, primals_66, buf71, primals_61, buf73, buf74, 384, 1568, grid=grid(384, 1568), stream=stream0)
        del primals_65
        buf75 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_139, (384, 1536), (1, 384), 0), out=buf75)
        buf76 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_86, x_89], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf75, primals_140, buf76, 2408448, grid=grid(2408448), stream=stream0)
        buf77 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_142, buf76, reinterpret_tensor(primals_141, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf77)
        del primals_142
        buf78 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_20, mul_21, x_84, x_91], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf71, primals_61, buf73, primals_64, buf77, buf78, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf79 = reinterpret_tensor(buf71, (3072, 196), (196, 1), 0); del buf71  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___linear_tokens], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(primals_68, primals_69, buf78, buf79, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_68
        buf80 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___linear_tokens], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf79, reinterpret_tensor(primals_143, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf80)
        del primals_144
        buf81 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul_23, mul_22, x_92, x_93], Original ATen: [aten._unsafe_view, aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused__unsafe_view_add_addcmul_clone_mul_9.run(primals_71, primals_72, buf78, primals_67, buf80, buf81, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_71
        buf82 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.mm]
        extern_kernels.mm(buf81, reinterpret_tensor(primals_145, (384, 1536), (1, 384), 0), out=buf82)
        buf83 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_94, x_97], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_2.run(buf82, primals_146, buf83, 2408448, grid=grid(2408448), stream=stream0)
        buf84 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, buf83, reinterpret_tensor(primals_147, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf84)
        del primals_148
        buf85 = empty_strided((8, 384, 2), (768, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, mul_23, x_100, x_102, x_103, x_92], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
        triton_red_fused_add_addcmul_mean_mul_10.run(primals_73, primals_74, buf78, primals_67, buf80, primals_70, buf84, buf85, 6144, 98, grid=grid(6144), stream=stream0)
        del buf78
        del primals_73
        buf86 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Source Nodes: [mul_22, mul_23, x_100, x_102, x_103, x_92], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
        triton_per_fused_add_addcmul_mean_mul_11.run(buf87, buf85, 3072, 2, grid=grid(3072), stream=stream0)
        del buf85
        buf88 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf87, reinterpret_tensor(primals_149, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf88)
        del primals_150
        return (buf88, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf9, buf10, buf11, buf12, buf13, buf14, buf16, buf17, buf18, buf19, buf20, buf21, buf23, buf24, buf25, buf26, buf27, buf28, buf30, buf31, buf32, buf33, buf34, buf35, buf37, buf38, buf39, buf40, buf41, buf42, buf44, buf45, buf46, buf47, buf48, buf49, buf51, buf52, buf53, buf54, buf55, buf56, buf58, buf59, buf60, buf61, buf62, buf63, buf65, buf66, buf67, buf68, buf69, buf70, buf72, buf73, buf74, buf75, buf76, buf77, buf79, buf80, buf81, buf82, buf83, buf84, buf87, reinterpret_tensor(primals_149, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_147, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_145, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_143, (196, 196), (196, 1), 0), reinterpret_tensor(primals_141, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_139, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_137, (196, 196), (196, 1), 0), reinterpret_tensor(primals_135, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_133, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_131, (196, 196), (196, 1), 0), reinterpret_tensor(primals_129, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_127, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_125, (196, 196), (196, 1), 0), reinterpret_tensor(primals_123, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_121, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_119, (196, 196), (196, 1), 0), reinterpret_tensor(primals_117, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_115, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_113, (196, 196), (196, 1), 0), reinterpret_tensor(primals_111, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_109, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_107, (196, 196), (196, 1), 0), reinterpret_tensor(primals_105, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_103, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_101, (196, 196), (196, 1), 0), reinterpret_tensor(primals_99, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_97, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_95, (196, 196), (196, 1), 0), reinterpret_tensor(primals_93, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_91, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_89, (196, 196), (196, 1), 0), reinterpret_tensor(primals_87, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_85, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_83, (196, 196), (196, 1), 0), reinterpret_tensor(primals_81, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_79, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_77, (196, 196), (196, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
