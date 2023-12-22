
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


# kernel path: /tmp/torchinductor_youkaichao/hk/chksgeakrgt5nyvljzeqse2a3f7ek5szzn6tjsalyfg4u6xeapjk.py
# Source Nodes: [addcmul], Original ATen: [aten.addcmul]
# addcmul => add, mul, mul_1
triton_poi_fused_addcmul_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addcmul_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/hw/chwny7he5ktmeimd6udmwtxsq22ifnwkvt3c3l5aclyoz3x4beoh.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((384*x1) + (75264*(x0 // 384)) + (x0 % 384)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvxvhx3vcqhwmoryxkehmn4ygjhiuou74y3yg6t4rovjz6j4cxz.py
# Source Nodes: [mul, x_4], Original ATen: [aten.add, aten.mul]
# mul => mul_2
# x_4 => add_1
triton_poi_fused_add_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    x0 = xindex % 196
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfml34dipujqa255va4yprtbz2ibomhs55jpkvkxr345hrsf7jqv.py
# Source Nodes: [addcmul_1, x_5], Original ATen: [aten.addcmul, aten.clone]
# addcmul_1 => add_2, mul_3, mul_4
# x_5 => clone
triton_poi_fused_addcmul_clone_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addcmul_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdtamw34hvdxocmp4gvd7skawcb5ium6cvpngyvawstwe26vk7k.py
# Source Nodes: [x_5, x_6], Original ATen: [aten.add, aten.gelu]
# x_5 => add_3
# x_6 => add_4, erf, mul_5, mul_6, mul_7
triton_poi_fused_add_gelu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhdvu4nkihopd64vij4c72znwexgdtostapnuo2mm4cfsuim3jw.py
# Source Nodes: [addcmul_2, mul_1, x_11], Original ATen: [aten.add, aten.addcmul, aten.mul]
# addcmul_2 => add_6, mul_10, mul_9
# mul_1 => mul_8
# x_11 => add_5
triton_poi_fused_add_addcmul_mul_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_mul_5', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp4 + tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6keblj2r4g3srx34ijvux2lzjza6k6pzlzhtuf44jfnaf6r47qm.py
# Source Nodes: [addcmul_3, mul_1, mul_2, x_11, x_12, x_13], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
# addcmul_3 => add_8, mul_12, mul_13
# mul_1 => mul_8
# mul_2 => mul_11
# x_11 => add_5
# x_12 => add_7
# x_13 => clone_3
triton_poi_fused_add_addcmul_clone_mul_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_clone_mul_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tmp13 + tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp12, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75wqg5meazbmrd5bth3qcuxfraly35piqsluk2xs7yhzch5soth.py
# Source Nodes: [mul_3, mul_4, x_19, x_20], Original ATen: [aten.add, aten.mul]
# mul_3 => mul_17
# mul_4 => mul_20
# x_19 => add_11
# x_20 => add_13
triton_poi_fused_add_mul_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhl5uec5epi7ik3mqo6ytrhjapstf2u3g536jvpwk64rrk25sz6.py
# Source Nodes: [addcmul_7, mul_5, mul_6, x_27, x_28, x_29], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
# addcmul_7 => add_20, mul_30, mul_31
# mul_5 => mul_26
# mul_6 => mul_29
# x_27 => add_17
# x_28 => add_19
# x_29 => clone_9
triton_poi_fused_add_addcmul_clone_mul_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_clone_mul_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tmp13 + tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp12, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43vhvkhagb4ous345p7zm5z5jiuk24w54dzkqjg2mayqoihw6fx.py
# Source Nodes: [addcmul_23, mul_21, mul_22, x_91, x_92, x_93], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
# addcmul_23 => add_68, mul_102, mul_103
# mul_21 => mul_98
# mul_22 => mul_101
# x_91 => add_65
# x_92 => add_67
# x_93 => clone_33
triton_poi_fused_add_addcmul_clone_mul_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_clone_mul_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tmp13 + tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp12, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czk4sow6hmzreqzk67jebvebul3ks4dpgeulbx7whnc3fhv66hjp.py
# Source Nodes: [mul_23, x_100, x_102, x_103], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
# mul_23 => mul_107
# x_100 => add_71
# x_102 => add_72, mul_108, mul_109
# x_103 => mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addcmul_mean_mul_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    x1 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r2) + (37632*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (384*r2) + (37632*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp3 * tmp10
        tmp12 = tmp0 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3x3uc2ss5vpuswwc4trgktd6mlwuh72mtmd4v4kigxwvwctkbc.py
# Source Nodes: [mul_23, x_100, x_102, x_103], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
# mul_23 => mul_107
# x_100 => add_71
# x_102 => add_72, mul_108, mul_109
# x_103 => mean
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (384, ), (1, ))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg5_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg8_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg11_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg12_1, (384, ), (1, ))
    assert_size_stride(arg13_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg14_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg17_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg20_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg21_1, (384, ), (1, ))
    assert_size_stride(arg22_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg23_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg24_1, (384, ), (1, ))
    assert_size_stride(arg25_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg26_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg29_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg30_1, (384, ), (1, ))
    assert_size_stride(arg31_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg32_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg35_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg36_1, (384, ), (1, ))
    assert_size_stride(arg37_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg38_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg41_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg44_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg47_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg50_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg53_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg56_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg59_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg62_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg65_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg68_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg71_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg72_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg73_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg74_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (196, 196), (196, 1))
    assert_size_stride(arg77_1, (196, ), (1, ))
    assert_size_stride(arg78_1, (1536, 384), (384, 1))
    assert_size_stride(arg79_1, (1536, ), (1, ))
    assert_size_stride(arg80_1, (384, 1536), (1536, 1))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (196, 196), (196, 1))
    assert_size_stride(arg83_1, (196, ), (1, ))
    assert_size_stride(arg84_1, (1536, 384), (384, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (384, 1536), (1536, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (196, 196), (196, 1))
    assert_size_stride(arg89_1, (196, ), (1, ))
    assert_size_stride(arg90_1, (1536, 384), (384, 1))
    assert_size_stride(arg91_1, (1536, ), (1, ))
    assert_size_stride(arg92_1, (384, 1536), (1536, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (196, 196), (196, 1))
    assert_size_stride(arg95_1, (196, ), (1, ))
    assert_size_stride(arg96_1, (1536, 384), (384, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (384, 1536), (1536, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (196, 196), (196, 1))
    assert_size_stride(arg101_1, (196, ), (1, ))
    assert_size_stride(arg102_1, (1536, 384), (384, 1))
    assert_size_stride(arg103_1, (1536, ), (1, ))
    assert_size_stride(arg104_1, (384, 1536), (1536, 1))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (196, 196), (196, 1))
    assert_size_stride(arg107_1, (196, ), (1, ))
    assert_size_stride(arg108_1, (1536, 384), (384, 1))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (384, 1536), (1536, 1))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (196, 196), (196, 1))
    assert_size_stride(arg113_1, (196, ), (1, ))
    assert_size_stride(arg114_1, (1536, 384), (384, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (384, 1536), (1536, 1))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (196, 196), (196, 1))
    assert_size_stride(arg119_1, (196, ), (1, ))
    assert_size_stride(arg120_1, (1536, 384), (384, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (384, 1536), (1536, 1))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (196, 196), (196, 1))
    assert_size_stride(arg125_1, (196, ), (1, ))
    assert_size_stride(arg126_1, (1536, 384), (384, 1))
    assert_size_stride(arg127_1, (1536, ), (1, ))
    assert_size_stride(arg128_1, (384, 1536), (1536, 1))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (196, 196), (196, 1))
    assert_size_stride(arg131_1, (196, ), (1, ))
    assert_size_stride(arg132_1, (1536, 384), (384, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (384, 1536), (1536, 1))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (196, 196), (196, 1))
    assert_size_stride(arg137_1, (196, ), (1, ))
    assert_size_stride(arg138_1, (1536, 384), (384, 1))
    assert_size_stride(arg139_1, (1536, ), (1, ))
    assert_size_stride(arg140_1, (384, 1536), (1536, 1))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (196, 196), (196, 1))
    assert_size_stride(arg143_1, (196, ), (1, ))
    assert_size_stride(arg144_1, (1536, 384), (384, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (384, 1536), (1536, 1))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (1000, 384), (384, 1))
    assert_size_stride(arg149_1, (1000, ), (1, ))
    assert_size_stride(arg150_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg150_1, arg74_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg150_1
        del arg74_1
        buf1 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [addcmul], Original ATen: [aten.addcmul]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_addcmul_0.run(arg1_1, arg2_1, buf0, arg75_1, buf1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg1_1
        del arg2_1
        buf2 = empty_strided((3072, 196), (1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf1, buf2, 602112, grid=grid(602112), stream=stream0)
        buf3 = reinterpret_tensor(buf1, (3072, 196), (196, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf2, reinterpret_tensor(arg76_1, (196, 196), (1, 196), 0), out=buf3)
        del arg76_1
        buf4 = reinterpret_tensor(buf0, (8, 196, 384), (75264, 1, 196), 0); del buf0  # reuse
        # Source Nodes: [mul, x_4], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf4, arg75_1, arg0_1, buf3, arg77_1, 602112, grid=grid(602112), stream=stream0)
        del arg0_1
        del arg75_1
        del arg77_1
        buf5 = reinterpret_tensor(buf3, (8, 196, 384), (75264, 384, 1), 0); del buf3  # reuse
        # Source Nodes: [addcmul_1, x_5], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg4_1, arg5_1, buf4, buf5, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg4_1
        del arg5_1
        buf6 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1568, 384), (384, 1), 0), reinterpret_tensor(arg78_1, (384, 1536), (1, 384), 0), out=buf6)
        del arg78_1
        buf7 = reinterpret_tensor(buf6, (8, 196, 1536), (301056, 1536, 1), 0); del buf6  # reuse
        # Source Nodes: [x_5, x_6], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf7, arg79_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg79_1
        buf8 = reinterpret_tensor(buf5, (1568, 384), (384, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg80_1, (1536, 384), (1, 1536), 0), out=buf8)
        del arg80_1
        buf9 = reinterpret_tensor(buf2, (8, 196, 384), (75264, 384, 1), 0); del buf2  # reuse
        # Source Nodes: [addcmul_2, mul_1, x_11], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg7_1, arg8_1, buf4, arg3_1, buf8, arg81_1, buf9, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg7_1
        del arg8_1
        buf10 = empty_strided((3072, 196), (1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf9, buf10, 602112, grid=grid(602112), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (3072, 196), (196, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, reinterpret_tensor(arg82_1, (196, 196), (1, 196), 0), out=buf11)
        del arg82_1
        buf12 = reinterpret_tensor(buf11, (8, 196, 384), (75264, 1, 196), 0); del buf11  # reuse
        buf13 = reinterpret_tensor(buf10, (8, 196, 384), (75264, 384, 1), 0); del buf10  # reuse
        # Source Nodes: [addcmul_3, mul_1, mul_2, x_11, x_12, x_13], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_6.run(buf12, buf4, arg3_1, buf8, arg81_1, arg6_1, arg83_1, arg10_1, arg11_1, buf13, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg10_1
        del arg11_1
        del arg3_1
        del arg6_1
        del arg81_1
        del arg83_1
        buf14 = reinterpret_tensor(buf7, (1568, 1536), (1536, 1), 0); del buf7  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1568, 384), (384, 1), 0), reinterpret_tensor(arg84_1, (384, 1536), (1, 384), 0), out=buf14)
        del arg84_1
        buf15 = reinterpret_tensor(buf14, (8, 196, 1536), (301056, 1536, 1), 0); del buf14  # reuse
        # Source Nodes: [x_13, x_14], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf15, arg85_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg85_1
        buf16 = reinterpret_tensor(buf13, (1568, 384), (384, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg86_1, (1536, 384), (1, 1536), 0), out=buf16)
        del arg86_1
        buf17 = reinterpret_tensor(buf8, (8, 196, 384), (75264, 384, 1), 0); del buf8  # reuse
        # Source Nodes: [addcmul_4, mul_3, x_19], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg13_1, arg14_1, buf12, arg9_1, buf16, arg87_1, buf17, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg13_1
        del arg14_1
        buf18 = reinterpret_tensor(buf4, (3072, 196), (1, 3072), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf17, buf18, 602112, grid=grid(602112), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (3072, 196), (196, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg88_1, (196, 196), (1, 196), 0), out=buf19)
        del arg88_1
        buf20 = buf12; del buf12  # reuse
        # Source Nodes: [mul_3, mul_4, x_19, x_20], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf20, arg9_1, buf16, arg87_1, arg12_1, buf19, arg89_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg12_1
        del arg87_1
        del arg89_1
        del arg9_1
        buf21 = reinterpret_tensor(buf19, (8, 196, 384), (75264, 384, 1), 0); del buf19  # reuse
        # Source Nodes: [addcmul_5, x_21], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg16_1, arg17_1, buf20, buf21, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg16_1
        del arg17_1
        buf22 = reinterpret_tensor(buf15, (1568, 1536), (1536, 1), 0); del buf15  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (1568, 384), (384, 1), 0), reinterpret_tensor(arg90_1, (384, 1536), (1, 384), 0), out=buf22)
        del arg90_1
        buf23 = reinterpret_tensor(buf22, (8, 196, 1536), (301056, 1536, 1), 0); del buf22  # reuse
        # Source Nodes: [x_21, x_22], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf23, arg91_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg91_1
        buf24 = reinterpret_tensor(buf21, (1568, 384), (384, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg92_1, (1536, 384), (1, 1536), 0), out=buf24)
        del arg92_1
        buf25 = reinterpret_tensor(buf16, (8, 196, 384), (75264, 384, 1), 0); del buf16  # reuse
        # Source Nodes: [addcmul_6, mul_5, x_27], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg19_1, arg20_1, buf20, arg15_1, buf24, arg93_1, buf25, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg19_1
        del arg20_1
        buf26 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf25, buf26, 602112, grid=grid(602112), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (3072, 196), (196, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(arg94_1, (196, 196), (1, 196), 0), out=buf27)
        del arg94_1
        buf28 = buf20; del buf20  # reuse
        buf29 = reinterpret_tensor(buf26, (8, 196, 384), (75264, 384, 1), 0); del buf26  # reuse
        # Source Nodes: [addcmul_7, mul_5, mul_6, x_27, x_28, x_29], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_8.run(buf28, arg15_1, buf24, arg93_1, arg18_1, buf27, arg95_1, arg22_1, arg23_1, buf29, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg15_1
        del arg18_1
        del arg22_1
        del arg23_1
        del arg93_1
        del arg95_1
        buf30 = reinterpret_tensor(buf23, (1568, 1536), (1536, 1), 0); del buf23  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1568, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 1536), (1, 384), 0), out=buf30)
        del arg96_1
        buf31 = reinterpret_tensor(buf30, (8, 196, 1536), (301056, 1536, 1), 0); del buf30  # reuse
        # Source Nodes: [x_29, x_30], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf31, arg97_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg97_1
        buf32 = reinterpret_tensor(buf29, (1568, 384), (384, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg98_1, (1536, 384), (1, 1536), 0), out=buf32)
        del arg98_1
        buf33 = reinterpret_tensor(buf27, (8, 196, 384), (75264, 384, 1), 0); del buf27  # reuse
        # Source Nodes: [addcmul_8, mul_7, x_35], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg25_1, arg26_1, buf28, arg21_1, buf32, arg99_1, buf33, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg25_1
        del arg26_1
        buf34 = reinterpret_tensor(buf24, (3072, 196), (1, 3072), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf33, buf34, 602112, grid=grid(602112), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (3072, 196), (196, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(arg100_1, (196, 196), (1, 196), 0), out=buf35)
        del arg100_1
        buf36 = buf28; del buf28  # reuse
        # Source Nodes: [mul_7, mul_8, x_35, x_36], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf36, arg21_1, buf32, arg99_1, arg24_1, buf35, arg101_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg101_1
        del arg21_1
        del arg24_1
        del arg99_1
        buf37 = reinterpret_tensor(buf35, (8, 196, 384), (75264, 384, 1), 0); del buf35  # reuse
        # Source Nodes: [addcmul_9, x_37], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg28_1, arg29_1, buf36, buf37, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg28_1
        del arg29_1
        buf38 = reinterpret_tensor(buf31, (1568, 1536), (1536, 1), 0); del buf31  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (1568, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 1536), (1, 384), 0), out=buf38)
        del arg102_1
        buf39 = reinterpret_tensor(buf38, (8, 196, 1536), (301056, 1536, 1), 0); del buf38  # reuse
        # Source Nodes: [x_37, x_38], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf39, arg103_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg103_1
        buf40 = reinterpret_tensor(buf37, (1568, 384), (384, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg104_1, (1536, 384), (1, 1536), 0), out=buf40)
        del arg104_1
        buf41 = reinterpret_tensor(buf32, (8, 196, 384), (75264, 384, 1), 0); del buf32  # reuse
        # Source Nodes: [addcmul_10, mul_9, x_43], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg31_1, arg32_1, buf36, arg27_1, buf40, arg105_1, buf41, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg31_1
        del arg32_1
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf41, buf42, 602112, grid=grid(602112), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (3072, 196), (196, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(arg106_1, (196, 196), (1, 196), 0), out=buf43)
        del arg106_1
        buf44 = buf36; del buf36  # reuse
        buf45 = reinterpret_tensor(buf42, (8, 196, 384), (75264, 384, 1), 0); del buf42  # reuse
        # Source Nodes: [addcmul_11, mul_10, mul_9, x_43, x_44, x_45], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_8.run(buf44, arg27_1, buf40, arg105_1, arg30_1, buf43, arg107_1, arg34_1, arg35_1, buf45, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg105_1
        del arg107_1
        del arg27_1
        del arg30_1
        del arg34_1
        del arg35_1
        buf46 = reinterpret_tensor(buf39, (1568, 1536), (1536, 1), 0); del buf39  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1568, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 1536), (1, 384), 0), out=buf46)
        del arg108_1
        buf47 = reinterpret_tensor(buf46, (8, 196, 1536), (301056, 1536, 1), 0); del buf46  # reuse
        # Source Nodes: [x_45, x_46], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf47, arg109_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg109_1
        buf48 = reinterpret_tensor(buf45, (1568, 384), (384, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg110_1, (1536, 384), (1, 1536), 0), out=buf48)
        del arg110_1
        buf49 = reinterpret_tensor(buf43, (8, 196, 384), (75264, 384, 1), 0); del buf43  # reuse
        # Source Nodes: [addcmul_12, mul_11, x_51], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg37_1, arg38_1, buf44, arg33_1, buf48, arg111_1, buf49, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg37_1
        del arg38_1
        buf50 = reinterpret_tensor(buf40, (3072, 196), (1, 3072), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf49, buf50, 602112, grid=grid(602112), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (3072, 196), (196, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf50, reinterpret_tensor(arg112_1, (196, 196), (1, 196), 0), out=buf51)
        del arg112_1
        buf52 = buf44; del buf44  # reuse
        # Source Nodes: [mul_11, mul_12, x_51, x_52], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf52, arg33_1, buf48, arg111_1, arg36_1, buf51, arg113_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg111_1
        del arg113_1
        del arg33_1
        del arg36_1
        buf53 = reinterpret_tensor(buf51, (8, 196, 384), (75264, 384, 1), 0); del buf51  # reuse
        # Source Nodes: [addcmul_13, x_53], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg40_1, arg41_1, buf52, buf53, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg40_1
        del arg41_1
        buf54 = reinterpret_tensor(buf47, (1568, 1536), (1536, 1), 0); del buf47  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1568, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 1536), (1, 384), 0), out=buf54)
        del arg114_1
        buf55 = reinterpret_tensor(buf54, (8, 196, 1536), (301056, 1536, 1), 0); del buf54  # reuse
        # Source Nodes: [x_53, x_54], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf55, arg115_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg115_1
        buf56 = reinterpret_tensor(buf53, (1568, 384), (384, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg116_1, (1536, 384), (1, 1536), 0), out=buf56)
        del arg116_1
        buf57 = reinterpret_tensor(buf48, (8, 196, 384), (75264, 384, 1), 0); del buf48  # reuse
        # Source Nodes: [addcmul_14, mul_13, x_59], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg43_1, arg44_1, buf52, arg39_1, buf56, arg117_1, buf57, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg43_1
        del arg44_1
        buf58 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf57, buf58, 602112, grid=grid(602112), stream=stream0)
        buf59 = reinterpret_tensor(buf57, (3072, 196), (196, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf58, reinterpret_tensor(arg118_1, (196, 196), (1, 196), 0), out=buf59)
        del arg118_1
        buf60 = buf52; del buf52  # reuse
        buf61 = reinterpret_tensor(buf58, (8, 196, 384), (75264, 384, 1), 0); del buf58  # reuse
        # Source Nodes: [addcmul_15, mul_13, mul_14, x_59, x_60, x_61], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_8.run(buf60, arg39_1, buf56, arg117_1, arg42_1, buf59, arg119_1, arg46_1, arg47_1, buf61, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg117_1
        del arg119_1
        del arg39_1
        del arg42_1
        del arg46_1
        del arg47_1
        buf62 = reinterpret_tensor(buf55, (1568, 1536), (1536, 1), 0); del buf55  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1568, 384), (384, 1), 0), reinterpret_tensor(arg120_1, (384, 1536), (1, 384), 0), out=buf62)
        del arg120_1
        buf63 = reinterpret_tensor(buf62, (8, 196, 1536), (301056, 1536, 1), 0); del buf62  # reuse
        # Source Nodes: [x_61, x_62], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf63, arg121_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg121_1
        buf64 = reinterpret_tensor(buf61, (1568, 384), (384, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg122_1, (1536, 384), (1, 1536), 0), out=buf64)
        del arg122_1
        buf65 = reinterpret_tensor(buf59, (8, 196, 384), (75264, 384, 1), 0); del buf59  # reuse
        # Source Nodes: [addcmul_16, mul_15, x_67], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg49_1, arg50_1, buf60, arg45_1, buf64, arg123_1, buf65, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg49_1
        del arg50_1
        buf66 = reinterpret_tensor(buf56, (3072, 196), (1, 3072), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf65, buf66, 602112, grid=grid(602112), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (3072, 196), (196, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf66, reinterpret_tensor(arg124_1, (196, 196), (1, 196), 0), out=buf67)
        del arg124_1
        buf68 = buf60; del buf60  # reuse
        # Source Nodes: [mul_15, mul_16, x_67, x_68], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf68, arg45_1, buf64, arg123_1, arg48_1, buf67, arg125_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg123_1
        del arg125_1
        del arg45_1
        del arg48_1
        buf69 = reinterpret_tensor(buf67, (8, 196, 384), (75264, 384, 1), 0); del buf67  # reuse
        # Source Nodes: [addcmul_17, x_69], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg52_1, arg53_1, buf68, buf69, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg52_1
        del arg53_1
        buf70 = reinterpret_tensor(buf63, (1568, 1536), (1536, 1), 0); del buf63  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1568, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 1536), (1, 384), 0), out=buf70)
        del arg126_1
        buf71 = reinterpret_tensor(buf70, (8, 196, 1536), (301056, 1536, 1), 0); del buf70  # reuse
        # Source Nodes: [x_69, x_70], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf71, arg127_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg127_1
        buf72 = reinterpret_tensor(buf69, (1568, 384), (384, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg128_1, (1536, 384), (1, 1536), 0), out=buf72)
        del arg128_1
        buf73 = reinterpret_tensor(buf64, (8, 196, 384), (75264, 384, 1), 0); del buf64  # reuse
        # Source Nodes: [addcmul_18, mul_17, x_75], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg55_1, arg56_1, buf68, arg51_1, buf72, arg129_1, buf73, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg55_1
        del arg56_1
        buf74 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf73, buf74, 602112, grid=grid(602112), stream=stream0)
        buf75 = reinterpret_tensor(buf73, (3072, 196), (196, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf74, reinterpret_tensor(arg130_1, (196, 196), (1, 196), 0), out=buf75)
        del arg130_1
        buf76 = buf68; del buf68  # reuse
        buf77 = reinterpret_tensor(buf74, (8, 196, 384), (75264, 384, 1), 0); del buf74  # reuse
        # Source Nodes: [addcmul_19, mul_17, mul_18, x_75, x_76, x_77], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_8.run(buf76, arg51_1, buf72, arg129_1, arg54_1, buf75, arg131_1, arg58_1, arg59_1, buf77, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg129_1
        del arg131_1
        del arg51_1
        del arg54_1
        del arg58_1
        del arg59_1
        buf78 = reinterpret_tensor(buf71, (1568, 1536), (1536, 1), 0); del buf71  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1568, 384), (384, 1), 0), reinterpret_tensor(arg132_1, (384, 1536), (1, 384), 0), out=buf78)
        del arg132_1
        buf79 = reinterpret_tensor(buf78, (8, 196, 1536), (301056, 1536, 1), 0); del buf78  # reuse
        # Source Nodes: [x_77, x_78], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf79, arg133_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg133_1
        buf80 = reinterpret_tensor(buf77, (1568, 384), (384, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg134_1, (1536, 384), (1, 1536), 0), out=buf80)
        del arg134_1
        buf81 = reinterpret_tensor(buf75, (8, 196, 384), (75264, 384, 1), 0); del buf75  # reuse
        # Source Nodes: [addcmul_20, mul_19, x_83], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg61_1, arg62_1, buf76, arg57_1, buf80, arg135_1, buf81, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg61_1
        del arg62_1
        buf82 = reinterpret_tensor(buf72, (3072, 196), (1, 3072), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf81, buf82, 602112, grid=grid(602112), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (3072, 196), (196, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(arg136_1, (196, 196), (1, 196), 0), out=buf83)
        del arg136_1
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [mul_19, mul_20, x_83, x_84], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf84, arg57_1, buf80, arg135_1, arg60_1, buf83, arg137_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg135_1
        del arg137_1
        del arg57_1
        del arg60_1
        buf85 = reinterpret_tensor(buf83, (8, 196, 384), (75264, 384, 1), 0); del buf83  # reuse
        # Source Nodes: [addcmul_21, x_85], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg64_1, arg65_1, buf84, buf85, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg64_1
        del arg65_1
        buf86 = reinterpret_tensor(buf79, (1568, 1536), (1536, 1), 0); del buf79  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 384), (384, 1), 0), reinterpret_tensor(arg138_1, (384, 1536), (1, 384), 0), out=buf86)
        del arg138_1
        buf87 = reinterpret_tensor(buf86, (8, 196, 1536), (301056, 1536, 1), 0); del buf86  # reuse
        # Source Nodes: [x_85, x_86], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf87, arg139_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg139_1
        buf88 = reinterpret_tensor(buf85, (1568, 384), (384, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg140_1, (1536, 384), (1, 1536), 0), out=buf88)
        del arg140_1
        buf89 = reinterpret_tensor(buf80, (8, 196, 384), (75264, 384, 1), 0); del buf80  # reuse
        # Source Nodes: [addcmul_22, mul_21, x_91], Original ATen: [aten.add, aten.addcmul, aten.mul]
        triton_poi_fused_add_addcmul_mul_5.run(arg67_1, arg68_1, buf84, arg63_1, buf88, arg141_1, buf89, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg67_1
        del arg68_1
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf89, buf90, 602112, grid=grid(602112), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (3072, 196), (196, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf90, reinterpret_tensor(arg142_1, (196, 196), (1, 196), 0), out=buf91)
        del arg142_1
        buf92 = reinterpret_tensor(buf88, (8, 196, 384), (75264, 384, 1), 0); del buf88  # reuse
        buf93 = reinterpret_tensor(buf90, (8, 196, 384), (75264, 384, 1), 0); del buf90  # reuse
        # Source Nodes: [addcmul_23, mul_21, mul_22, x_91, x_92, x_93], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
        triton_poi_fused_add_addcmul_clone_mul_9.run(buf92, buf84, arg63_1, arg141_1, arg66_1, buf91, arg143_1, arg70_1, arg71_1, buf93, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg141_1
        del arg143_1
        del arg63_1
        del arg66_1
        del arg70_1
        del arg71_1
        del buf84
        del buf91
        buf94 = reinterpret_tensor(buf87, (1568, 1536), (1536, 1), 0); del buf87  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1568, 384), (384, 1), 0), reinterpret_tensor(arg144_1, (384, 1536), (1, 384), 0), out=buf94)
        del arg144_1
        buf95 = reinterpret_tensor(buf94, (8, 196, 1536), (301056, 1536, 1), 0); del buf94  # reuse
        # Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf95, arg145_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg145_1
        buf96 = reinterpret_tensor(buf93, (1568, 384), (384, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg146_1, (1536, 384), (1, 1536), 0), out=buf96)
        del arg146_1
        del buf95
        buf97 = empty_strided((8, 384, 2), (768, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_23, x_100, x_102, x_103], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
        triton_red_fused_add_addcmul_mean_mul_10.run(arg72_1, arg73_1, buf92, arg69_1, buf96, arg147_1, buf97, 6144, 98, grid=grid(6144), stream=stream0)
        del arg147_1
        del arg69_1
        del arg72_1
        del arg73_1
        del buf92
        del buf96
        buf98 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf99 = buf98; del buf98  # reuse
        # Source Nodes: [mul_23, x_100, x_102, x_103], Original ATen: [aten.add, aten.addcmul, aten.mean, aten.mul]
        triton_per_fused_add_addcmul_mean_mul_11.run(buf99, buf97, 3072, 2, grid=grid(3072), stream=stream0)
        del buf97
        buf100 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_23, x_100, x_102, x_103, x_105], Original ATen: [aten.add, aten.addcmul, aten.addmm, aten.mean, aten.mul]
        extern_kernels.addmm(arg149_1, buf99, reinterpret_tensor(arg148_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf100)
        del arg148_1
        del arg149_1
        return (buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
