
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


# kernel path: /tmp/torchinductor_youkaichao/uy/cuy5zangda6xjnfzooy4hwrgp7mp7o43cuklpqec5xlcghpvltqz.py
# Source Nodes: [mul, mul_1, x_11, x_4], Original ATen: [aten.add, aten.mul]
# mul => mul_2
# mul_1 => mul_8
# x_11 => add_5
# x_4 => add_1
triton_poi_fused_add_mul_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': []},
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
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ywg36f46qua2fw6u7uwof2dee4gmpvlhupvj2a6m57lwcl6w42.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul]

triton_poi_fused_div_mul_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x2 = (xindex // 75264)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = 196.0
    tmp2 = tmp0 / tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cliltojiawrjc3ix2hsozpyofnohd3mw2m3wmgeu2jmipap57t4w.py
# Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
# x_93 => add_69
# x_94 => add_70, erf_11, mul_105
triton_poi_fused_add_gelu_gelu_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp0 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjv7e657msy6xmb7l4vjrved2urgkkapwnumxxuvedengr5w3sd.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 384
    x2 = xindex
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp1 = 196.0
    tmp2 = tmp0 / tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp8 * tmp4
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7gaswtyolpe3i2i6sldsfvitkdfm55d7wbgiyinxrvskqflt4c.py
# Source Nodes: [mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, x_12, x_19, x_20, x_27, x_28, x_35, x_36, x_43, x_44, x_51, x_52, x_59, x_60, x_67, x_68, x_75, x_76, x_83, x_84, x_91], Original ATen: [aten.add, aten.div, aten.mul]
# mul_10 => mul_47
# mul_11 => mul_53
# mul_12 => mul_56
# mul_13 => mul_62
# mul_14 => mul_65
# mul_15 => mul_71
# mul_16 => mul_74
# mul_17 => mul_80
# mul_18 => mul_83
# mul_19 => mul_89
# mul_2 => mul_11
# mul_20 => mul_92
# mul_21 => mul_98
# mul_3 => mul_17
# mul_4 => mul_20
# mul_5 => mul_26
# mul_6 => mul_29
# mul_7 => mul_35
# mul_8 => mul_38
# mul_9 => mul_44
# x_12 => add_7
# x_19 => add_11
# x_20 => add_13
# x_27 => add_17
# x_28 => add_19
# x_35 => add_23
# x_36 => add_25
# x_43 => add_29
# x_44 => add_31
# x_51 => add_35
# x_52 => add_37
# x_59 => add_41
# x_60 => add_43
# x_67 => add_47
# x_68 => add_49
# x_75 => add_53
# x_76 => add_55
# x_83 => add_59
# x_84 => add_61
# x_91 => add_65
triton_poi_fused_add_div_mul_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(59, 60))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr14 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr16 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr17 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr24 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr26 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr27 + (x2), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr28 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr29 + (x2), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr32 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr33 + (x2), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr34 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr35 + (x2), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr36 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr37 + (x2), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr38 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr39 + (x2), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr40 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr41 + (x2 + (384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr42 + (x2), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr43 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr44 + (x2), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr45 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr46 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp31 = tmp29 * tmp30
    tmp32 = tmp28 + tmp31
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 + tmp43
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 + tmp47
    tmp51 = tmp49 * tmp50
    tmp52 = tmp48 + tmp51
    tmp55 = tmp53 * tmp54
    tmp56 = tmp52 + tmp55
    tmp59 = tmp57 * tmp58
    tmp60 = tmp56 + tmp59
    tmp63 = tmp61 * tmp62
    tmp64 = tmp60 + tmp63
    tmp67 = tmp65 * tmp66
    tmp68 = tmp64 + tmp67
    tmp71 = tmp69 * tmp70
    tmp72 = tmp68 + tmp71
    tmp75 = tmp73 * tmp74
    tmp76 = tmp72 + tmp75
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp82 = 196.0
    tmp83 = tmp81 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp83 * tmp86
    tmp90 = tmp89 * tmp85
    tmp91 = tmp88 * tmp90
    tmp92 = tmp87 + tmp91
    tmp95 = tmp94 * tmp85
    tmp96 = tmp93 * tmp95
    tmp97 = tmp92 + tmp96
    tmp98 = tmp97 * tmp77
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp16, xmask & ymask)
    tl.store(out_ptr2 + (x2 + (384*y3)), tmp24, xmask & ymask)
    tl.store(out_ptr3 + (x2 + (384*y3)), tmp32, xmask & ymask)
    tl.store(out_ptr4 + (x2 + (384*y3)), tmp40, xmask & ymask)
    tl.store(out_ptr5 + (x2 + (384*y3)), tmp48, xmask & ymask)
    tl.store(out_ptr6 + (x2 + (384*y3)), tmp56, xmask & ymask)
    tl.store(out_ptr7 + (x2 + (384*y3)), tmp64, xmask & ymask)
    tl.store(out_ptr8 + (x2 + (384*y3)), tmp72, xmask & ymask)
    tl.store(out_ptr9 + (x2 + (384*y3)), tmp80, xmask & ymask)
    tl.store(out_ptr10 + (x2 + (384*y3)), tmp97, xmask & ymask)
    tl.store(out_ptr11 + (x2 + (384*y3)), tmp98, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy37xnfiowz3yniy3roo5mhfefgx3oa3rlbm7b2rtn5tcf2mw6hy.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czml6epurrirnatc2ivrsyoenci4gmj3gl4tlnlb3sdb7zcqa4k4.py
# Source Nodes: [], Original ATen: [aten.div, aten.sum]

triton_red_fused_div_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepznk4gzvvdaawrc5g7xvndvx7kmlnaqx5vjsxlc7xo4ufui352.py
# Source Nodes: [mul_22, mul_23, x_100, x_92], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# mul_22 => mul_101
# mul_23 => mul_107
# x_100 => add_71
# x_92 => add_67
triton_red_fused_add_div_mul_sum_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp43 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (384*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = 196.0
        tmp11 = tmp9 / tmp10
        tmp12 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr6 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp16 + tmp19
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp23 = tmp11 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp29 = tmp16 * tmp21
        tmp30 = tmp3 * tmp29
        tmp31 = tl.full(tmp30.shape, 0, tmp30.dtype)
        tmp32 = tl.where(tmp2, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
        tmp36 = tl.load(in_ptr7 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tmp36 * tmp21
        tmp38 = tmp11 * tmp37
        tmp39 = tmp38 * tmp18
        tmp40 = tl.full(tmp39.shape, 0, tmp39.dtype)
        tmp41 = tl.where(tmp2, tmp39, tmp40)
        tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
        tmp44 = _tmp43 + tmp42
        _tmp43 = tl.where(rmask & xmask, tmp44, _tmp43)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp34, xmask)
    tmp43 = tl.sum(_tmp43, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp43, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwjj6agccfaxe3js4ina2w5agkex37mkhk7dbvifxagohytk47r.py
# Source Nodes: [mul_22, mul_23, x_100, x_92], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# mul_22 => mul_101
# mul_23 => mul_107
# x_100 => add_71
# x_92 => add_67
triton_per_fused_add_div_mul_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3hbrbodc2gwbzz7pn6ze2rbpm7uqj2eryvy7jvvdriksq5yba5x.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtgctiafbqpmegxylndpvwrdh4vfvyfip37fifr3evqovuwriou.py
# Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
# x_93 => add_69
# x_94 => add_70, erf_11, mul_105
triton_red_fused_add_gelu_gelu_backward_sum_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19968
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x0 = xindex % 1536
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (1536*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 0.7071067811865476
        tmp8 = tmp6 * tmp7
        tmp9 = tl.math.erf(tmp8)
        tmp10 = 1.0
        tmp11 = tmp9 + tmp10
        tmp12 = 0.5
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp6
        tmp15 = -0.5
        tmp16 = tmp14 * tmp15
        tmp17 = tl.exp(tmp16)
        tmp18 = 0.3989422804014327
        tmp19 = tmp17 * tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tmp13 + tmp20
        tmp22 = tmp3 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqum56zt3u64slynnr7st4in5zbsahtdg6ioekkghrdpiutbvkkk.py
# Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
# x_93 => add_69
# x_94 => add_70, erf_11, mul_105
triton_per_fused_add_gelu_gelu_backward_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2i5ibe6qo5hvdoci4hlqnwhoz2oaebut4stzgqon4xo7c7yd3bd.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]

triton_red_fused_add_div_mul_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 196.0
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.load(in_ptr2 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp11 * tmp7
        tmp13 = tmp10 * tmp12
        tmp14 = tmp9 + tmp13
        tmp15 = tl.load(in_ptr4 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4iz7d6yjhtgxmfjwpbotkzmybmfxoy26pmy5ig5t4yuxjnrrmf.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]

triton_per_fused_add_div_mul_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3vvtbjrkxkhhlazedtd4xrg2ds5ahoh5ptq3yd77y5if7tpgur.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqnj7a2aycvxqsok5wkzufuketjsozsj3q6oz4riowphn72o6lk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuuamowul6b5tz6pqzwudzrkdb7h27k6fa2kpiqzph3mmybqzf2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cnikznxuvsrxywxtn2urqyxdhkn42n6yvz5bpl7zuy7bym5kxhot.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3kk4mt5piiuefpnz7tmralwuzumm32inwtxwnc2pczroj4ts3a.py
# Source Nodes: [mul_20, x_84], Original ATen: [aten.add, aten.mul, aten.sum]
# mul_20 => mul_92
# x_84 => add_61
triton_red_fused_add_mul_sum_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp17 = tl.load(in_ptr3 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr5 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp17 + tmp20
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp24 = tmp11 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tl.load(in_ptr6 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp30 * tmp22
        tmp32 = tmp11 * tmp31
        tmp33 = tmp3 + tmp32
        tmp34 = tmp33 * tmp19
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp2, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask & xmask, tmp39, _tmp38)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp28, xmask)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxedir6nzyvgaqihyuhbchfhfitof3pqbn5afvjpeky7zfqhldx.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xuvuomj3njmkss6e3gmakprjo2e2kk7ez6eccm2ahahgkuxiip.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp8 * tmp3
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp13 = tmp11 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiir2fohdfs6utysx2a22vguwvbjo7a644xzkql44s4y5fvqi3h.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp8 * tmp3
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp13 = tmp11 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckorpbdzrb62r34lttmoizlwhpqxsnv2yv67ik4ibmqps5cruca.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp11 * tmp19
        tmp21 = tmp3 + tmp20
        tmp22 = tl.load(in_ptr4 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqj5ca7ngti4kvw5pqhsg7zmfj65ki6edqcsahiobsvr6zvo7agu.py
# Source Nodes: [mul_2, x_12], Original ATen: [aten.add, aten.mul, aten.sum]
# mul_2 => mul_11
# x_12 => add_7
triton_red_fused_add_mul_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr3 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp1 + tmp4
        tmp6 = 1.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tmp1 * tmp6
        tmp17 = tmp12 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bni32zebnczdmztmc7pml5va7u2upd2xjxbjw6p43cbjyq5xze.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_24', 'mutated_arg_names': ['in_out_ptr0']},
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
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp8 * tmp3
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chuxr2fgshmte7dt4dgrqoh7y7msqpd6raumkrtkaehhnbmcw5zj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, convolution, view_1, addmm, view_3, mm, view_5, addmm_1, view_7, addmm_2, view_9, mm_1, view_11, addmm_3, view_13, addmm_4, view_15, mm_2, view_17, addmm_5, view_19, addmm_6, view_21, mm_3, view_23, addmm_7, view_25, addmm_8, view_27, mm_4, view_29, addmm_9, view_31, addmm_10, view_33, mm_5, view_35, addmm_11, view_37, addmm_12, view_39, mm_6, view_41, addmm_13, view_43, addmm_14, view_45, mm_7, view_47, addmm_15, view_49, addmm_16, view_51, mm_8, view_53, addmm_17, view_55, addmm_18, view_57, mm_9, view_59, addmm_19, view_61, addmm_20, view_63, mm_10, view_65, addmm_21, view_67, addmm_22, view_69, mm_11, view_71, addmm_23, clone_36, permute_62, permute_66, permute_72, permute_75, permute_80, permute_86, permute_89, permute_94, permute_100, permute_103, permute_108, permute_114, permute_117, permute_122, permute_128, permute_131, permute_136, permute_142, permute_145, permute_150, permute_156, permute_159, permute_164, permute_170, permute_173, permute_178, permute_184, permute_187, permute_192, permute_198, permute_201, permute_206, permute_212, permute_215, permute_220, permute_226, permute_229, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (384, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_6, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_9, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_12, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_13, (384, ), (1, ))
    assert_size_stride(primals_15, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_18, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_21, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_24, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_27, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_30, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_31, (384, ), (1, ))
    assert_size_stride(primals_33, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_36, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_39, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_42, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_48, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_54, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_60, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_66, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_69, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_72, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_74, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_75, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_80, (1536, ), (1, ))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_92, (1536, ), (1, ))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_128, (1536, ), (1, ))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(view_1, (3072, 196), (196, 1))
    assert_size_stride(addmm, (3072, 196), (196, 1))
    assert_size_stride(view_3, (1568, 384), (384, 1))
    assert_size_stride(mm, (1568, 1536), (1536, 1))
    assert_size_stride(view_5, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_1, (1568, 384), (384, 1))
    assert_size_stride(view_7, (3072, 196), (196, 1))
    assert_size_stride(addmm_2, (3072, 196), (196, 1))
    assert_size_stride(view_9, (1568, 384), (384, 1))
    assert_size_stride(mm_1, (1568, 1536), (1536, 1))
    assert_size_stride(view_11, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_3, (1568, 384), (384, 1))
    assert_size_stride(view_13, (3072, 196), (196, 1))
    assert_size_stride(addmm_4, (3072, 196), (196, 1))
    assert_size_stride(view_15, (1568, 384), (384, 1))
    assert_size_stride(mm_2, (1568, 1536), (1536, 1))
    assert_size_stride(view_17, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_5, (1568, 384), (384, 1))
    assert_size_stride(view_19, (3072, 196), (196, 1))
    assert_size_stride(addmm_6, (3072, 196), (196, 1))
    assert_size_stride(view_21, (1568, 384), (384, 1))
    assert_size_stride(mm_3, (1568, 1536), (1536, 1))
    assert_size_stride(view_23, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_7, (1568, 384), (384, 1))
    assert_size_stride(view_25, (3072, 196), (196, 1))
    assert_size_stride(addmm_8, (3072, 196), (196, 1))
    assert_size_stride(view_27, (1568, 384), (384, 1))
    assert_size_stride(mm_4, (1568, 1536), (1536, 1))
    assert_size_stride(view_29, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_9, (1568, 384), (384, 1))
    assert_size_stride(view_31, (3072, 196), (196, 1))
    assert_size_stride(addmm_10, (3072, 196), (196, 1))
    assert_size_stride(view_33, (1568, 384), (384, 1))
    assert_size_stride(mm_5, (1568, 1536), (1536, 1))
    assert_size_stride(view_35, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_11, (1568, 384), (384, 1))
    assert_size_stride(view_37, (3072, 196), (196, 1))
    assert_size_stride(addmm_12, (3072, 196), (196, 1))
    assert_size_stride(view_39, (1568, 384), (384, 1))
    assert_size_stride(mm_6, (1568, 1536), (1536, 1))
    assert_size_stride(view_41, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_13, (1568, 384), (384, 1))
    assert_size_stride(view_43, (3072, 196), (196, 1))
    assert_size_stride(addmm_14, (3072, 196), (196, 1))
    assert_size_stride(view_45, (1568, 384), (384, 1))
    assert_size_stride(mm_7, (1568, 1536), (1536, 1))
    assert_size_stride(view_47, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_15, (1568, 384), (384, 1))
    assert_size_stride(view_49, (3072, 196), (196, 1))
    assert_size_stride(addmm_16, (3072, 196), (196, 1))
    assert_size_stride(view_51, (1568, 384), (384, 1))
    assert_size_stride(mm_8, (1568, 1536), (1536, 1))
    assert_size_stride(view_53, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_17, (1568, 384), (384, 1))
    assert_size_stride(view_55, (3072, 196), (196, 1))
    assert_size_stride(addmm_18, (3072, 196), (196, 1))
    assert_size_stride(view_57, (1568, 384), (384, 1))
    assert_size_stride(mm_9, (1568, 1536), (1536, 1))
    assert_size_stride(view_59, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_19, (1568, 384), (384, 1))
    assert_size_stride(view_61, (3072, 196), (196, 1))
    assert_size_stride(addmm_20, (3072, 196), (196, 1))
    assert_size_stride(view_63, (1568, 384), (384, 1))
    assert_size_stride(mm_10, (1568, 1536), (1536, 1))
    assert_size_stride(view_65, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_21, (1568, 384), (384, 1))
    assert_size_stride(view_67, (3072, 196), (196, 1))
    assert_size_stride(addmm_22, (3072, 196), (196, 1))
    assert_size_stride(view_69, (1568, 384), (384, 1))
    assert_size_stride(mm_11, (1568, 1536), (1536, 1))
    assert_size_stride(view_71, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_23, (1568, 384), (384, 1))
    assert_size_stride(clone_36, (8, 384), (384, 1))
    assert_size_stride(permute_62, (1000, 384), (384, 1))
    assert_size_stride(permute_66, (384, 1536), (1536, 1))
    assert_size_stride(permute_72, (1536, 384), (384, 1))
    assert_size_stride(permute_75, (196, 196), (196, 1))
    assert_size_stride(permute_80, (384, 1536), (1536, 1))
    assert_size_stride(permute_86, (1536, 384), (384, 1))
    assert_size_stride(permute_89, (196, 196), (196, 1))
    assert_size_stride(permute_94, (384, 1536), (1536, 1))
    assert_size_stride(permute_100, (1536, 384), (384, 1))
    assert_size_stride(permute_103, (196, 196), (196, 1))
    assert_size_stride(permute_108, (384, 1536), (1536, 1))
    assert_size_stride(permute_114, (1536, 384), (384, 1))
    assert_size_stride(permute_117, (196, 196), (196, 1))
    assert_size_stride(permute_122, (384, 1536), (1536, 1))
    assert_size_stride(permute_128, (1536, 384), (384, 1))
    assert_size_stride(permute_131, (196, 196), (196, 1))
    assert_size_stride(permute_136, (384, 1536), (1536, 1))
    assert_size_stride(permute_142, (1536, 384), (384, 1))
    assert_size_stride(permute_145, (196, 196), (196, 1))
    assert_size_stride(permute_150, (384, 1536), (1536, 1))
    assert_size_stride(permute_156, (1536, 384), (384, 1))
    assert_size_stride(permute_159, (196, 196), (196, 1))
    assert_size_stride(permute_164, (384, 1536), (1536, 1))
    assert_size_stride(permute_170, (1536, 384), (384, 1))
    assert_size_stride(permute_173, (196, 196), (196, 1))
    assert_size_stride(permute_178, (384, 1536), (1536, 1))
    assert_size_stride(permute_184, (1536, 384), (384, 1))
    assert_size_stride(permute_187, (196, 196), (196, 1))
    assert_size_stride(permute_192, (384, 1536), (1536, 1))
    assert_size_stride(permute_198, (1536, 384), (384, 1))
    assert_size_stride(permute_201, (196, 196), (196, 1))
    assert_size_stride(permute_206, (384, 1536), (1536, 1))
    assert_size_stride(permute_212, (1536, 384), (384, 1))
    assert_size_stride(permute_215, (196, 196), (196, 1))
    assert_size_stride(permute_220, (384, 1536), (1536, 1))
    assert_size_stride(permute_226, (1536, 384), (384, 1))
    assert_size_stride(permute_229, (196, 196), (196, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 196, 384), (75264, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, mul_1, x_11, x_4], Original ATen: [aten.add, aten.mul]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_mul_0.run(convolution, primals_1, addmm, primals_4, addmm_1, buf0, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf11 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_62, out=buf11)
        del permute_62
        buf19 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_1.run(buf11, primals_74, primals_70, buf19, 602112, grid=grid(602112), stream=stream0)
        buf20 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (1568, 384), (384, 1), 0), permute_66, out=buf20)
        del permute_66
        buf26 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf20, mm_11, primals_146, buf26, 2408448, grid=grid(2408448), stream=stream0)
        buf28 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1568, 1536), (1536, 1), 0), permute_72, out=buf28)
        del permute_72
        buf35 = empty((8, 384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf11, primals_74, buf28, primals_72, primals_67, buf35, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf36 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (3072, 196), (196, 1), 0), permute_75, out=buf36)
        del permute_75
        buf1 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf2 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf3 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf4 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf5 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf6 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf7 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf8 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf9 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf10 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf43 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf46 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, x_12, x_19, x_20, x_27, x_28, x_35, x_36, x_43, x_44, x_51, x_52, x_59, x_60, x_67, x_68, x_75, x_76, x_83, x_84, x_91], Original ATen: [aten.add, aten.div, aten.mul]
        triton_poi_fused_add_div_mul_4.run(buf0, primals_7, addmm_2, primals_10, addmm_3, primals_13, addmm_4, primals_16, addmm_5, primals_19, addmm_6, primals_22, addmm_7, primals_25, addmm_8, primals_28, addmm_9, primals_31, addmm_10, primals_34, addmm_11, primals_37, addmm_12, primals_40, addmm_13, primals_43, addmm_14, primals_46, addmm_15, primals_49, addmm_16, primals_52, addmm_17, primals_55, addmm_18, primals_58, addmm_19, primals_61, addmm_20, primals_64, addmm_21, buf11, primals_74, buf28, primals_72, buf36, primals_69, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf43, buf46, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_64
        del primals_69
        buf12 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_36, out=buf12)
        del clone_36
        buf13 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(tangents_1, buf13, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf14 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.sum]
        triton_red_fused_div_sum_6.run(buf11, buf14, 384, 1568, grid=grid(384), stream=stream0)
        buf29 = empty_strided((1, 1, 384, 13), (4992, 4992, 1, 384), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 1, 384, 13), (4992, 4992, 1, 384), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 1, 384, 13), (4992, 4992, 1, 384), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((1, 1, 384, 13), (4992, 4992, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, mul_23, x_100, x_92], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_7.run(buf28, buf11, buf10, primals_67, addmm_22, primals_70, addmm_23, primals_74, buf29, buf15, buf31, buf17, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_23
        del primals_67
        del primals_70
        buf16 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, mul_23, x_100, x_92], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf15, buf16, 384, 13, grid=grid(384), stream=stream0)
        buf18 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf17, buf18, 384, 13, grid=grid(384), stream=stream0)
        buf21 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (384, 1568), (1, 384), 0), view_71, out=buf21)
        del view_71
        buf22 = reinterpret_tensor(buf17, (1, 384, 13), (4992, 1, 384), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf19, buf22, 4992, 121, grid=grid(4992), stream=stream0)
        del buf19
        buf23 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf22, buf23, 384, 13, grid=grid(384), stream=stream0)
        buf24 = empty_strided((1, 1, 1536, 13), (19968, 19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf20, mm_11, primals_146, buf24, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_11
        del primals_146
        buf25 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf24, buf25, 1536, 13, grid=grid(1536), stream=stream0)
        buf27 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1536, 1568), (1, 1536), 0), view_69, out=buf27)
        del view_69
        buf30 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf29, buf30, 384, 13, grid=grid(384), stream=stream0)
        buf32 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, x_92], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf31, buf32, 384, 13, grid=grid(384), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf11, primals_74, buf28, primals_72, addmm_22, buf33, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_22
        del buf11
        del buf28
        del primals_72
        del primals_74
        buf34 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf33, buf34, 384, 13, grid=grid(384), stream=stream0)
        buf37 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (196, 3072), (1, 196), 0), view_67, out=buf37)
        del view_67
        buf38 = empty_strided((1, 196, 24), (4704, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf35, buf38, 4704, 128, grid=grid(4704), stream=stream0)
        del buf35
        buf39 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf38, buf39, 196, 24, grid=grid(196), stream=stream0)
        buf40 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf36, buf40, 384, 1568, grid=grid(384), stream=stream0)
        buf41 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf36, buf10, buf41, 4992, 121, grid=grid(4992), stream=stream0)
        buf42 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf41, buf42, 384, 13, grid=grid(384), stream=stream0)
        buf47 = reinterpret_tensor(buf26, (1568, 1536), (1536, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (1568, 384), (384, 1), 0), permute_80, out=buf47)
        del permute_80
        buf53 = reinterpret_tensor(buf20, (8, 196, 1536), (301056, 1536, 1), 0); del buf20  # reuse
        # Source Nodes: [x_85, x_86], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf47, mm_10, primals_140, buf53, 2408448, grid=grid(2408448), stream=stream0)
        buf55 = reinterpret_tensor(buf36, (1568, 384), (384, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1568, 1536), (1536, 1), 0), permute_86, out=buf55)
        del permute_86
        buf44 = reinterpret_tensor(buf41, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf41  # reuse
        buf56 = buf29; del buf29  # reuse
        buf58 = reinterpret_tensor(buf22, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf22  # reuse
        buf60 = buf15; del buf15  # reuse
        # Source Nodes: [mul_20, x_84], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf43, addmm_21, buf55, buf9, primals_61, addmm_20, primals_66, buf44, buf56, buf58, buf60, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_20
        del addmm_21
        buf45 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf44, buf45, 384, 13, grid=grid(384), stream=stream0)
        buf48 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (384, 1568), (1, 384), 0), view_65, out=buf48)
        del view_65
        buf49 = reinterpret_tensor(buf44, (1, 384, 13), (4992, 1, 384), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf46, buf49, 4992, 121, grid=grid(4992), stream=stream0)
        buf50 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf49, buf50, 384, 13, grid=grid(384), stream=stream0)
        buf51 = buf24; del buf24  # reuse
        # Source Nodes: [x_85, x_86], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf47, mm_10, primals_140, buf51, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_10
        del primals_140
        buf52 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_86], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf51, buf52, 1536, 13, grid=grid(1536), stream=stream0)
        buf54 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1536, 1568), (1, 1536), 0), view_63, out=buf54)
        del view_63
        buf57 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf56, buf57, 384, 13, grid=grid(384), stream=stream0)
        buf59 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_20, x_84], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf58, buf59, 384, 13, grid=grid(384), stream=stream0)
        buf61 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf60, buf61, 384, 13, grid=grid(384), stream=stream0)
        buf62 = reinterpret_tensor(buf46, (8, 384, 196), (75264, 196, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf43, buf55, primals_66, primals_61, buf62, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_61
        buf63 = reinterpret_tensor(buf10, (3072, 196), (196, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (3072, 196), (196, 1), 0), permute_89, out=buf63)
        del permute_89
        buf64 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (196, 3072), (1, 196), 0), view_61, out=buf64)
        del view_61
        buf65 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf62, buf65, 4704, 128, grid=grid(4704), stream=stream0)
        del buf62
        buf66 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf65, buf66, 196, 24, grid=grid(196), stream=stream0)
        buf67 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf63, buf67, 384, 1568, grid=grid(384), stream=stream0)
        buf68 = reinterpret_tensor(buf60, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf63, buf9, buf68, 4992, 121, grid=grid(4992), stream=stream0)
        buf69 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf68, buf69, 384, 13, grid=grid(384), stream=stream0)
        buf70 = buf43; del buf43  # reuse
        buf73 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf70, buf55, primals_66, buf63, primals_63, primals_58, buf73, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_58
        del primals_63
        del primals_66
        buf74 = reinterpret_tensor(buf53, (1568, 1536), (1536, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (1568, 384), (384, 1), 0), permute_94, out=buf74)
        del permute_94
        buf80 = reinterpret_tensor(buf47, (8, 196, 1536), (301056, 1536, 1), 0); del buf47  # reuse
        # Source Nodes: [x_77, x_78], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf74, mm_9, primals_134, buf80, 2408448, grid=grid(2408448), stream=stream0)
        buf82 = reinterpret_tensor(buf63, (1568, 384), (384, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1568, 1536), (1536, 1), 0), permute_100, out=buf82)
        del permute_100
        buf71 = reinterpret_tensor(buf68, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf68  # reuse
        buf83 = buf58; del buf58  # reuse
        buf85 = buf56; del buf56  # reuse
        buf87 = reinterpret_tensor(buf49, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf49  # reuse
        # Source Nodes: [mul_18, x_76], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf70, addmm_19, buf82, buf8, primals_55, addmm_18, primals_60, buf71, buf83, buf85, buf87, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_18
        del addmm_19
        buf72 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf71, buf72, 384, 13, grid=grid(384), stream=stream0)
        buf75 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (384, 1568), (1, 384), 0), view_59, out=buf75)
        del view_59
        buf76 = reinterpret_tensor(buf71, (1, 384, 13), (4992, 1, 384), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf73, buf76, 4992, 121, grid=grid(4992), stream=stream0)
        buf77 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf76, buf77, 384, 13, grid=grid(384), stream=stream0)
        buf78 = buf51; del buf51  # reuse
        # Source Nodes: [x_77, x_78], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf74, mm_9, primals_134, buf78, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_9
        del primals_134
        buf79 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77, x_78], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf78, buf79, 1536, 13, grid=grid(1536), stream=stream0)
        buf81 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1536, 1568), (1, 1536), 0), view_57, out=buf81)
        del view_57
        buf84 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf83, buf84, 384, 13, grid=grid(384), stream=stream0)
        buf86 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_18, x_76], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf85, buf86, 384, 13, grid=grid(384), stream=stream0)
        buf88 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf87, buf88, 384, 13, grid=grid(384), stream=stream0)
        buf89 = reinterpret_tensor(buf73, (8, 384, 196), (75264, 196, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf70, buf82, primals_60, primals_55, buf89, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_55
        buf90 = reinterpret_tensor(buf55, (3072, 196), (196, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (3072, 196), (196, 1), 0), permute_103, out=buf90)
        del permute_103
        buf91 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (196, 3072), (1, 196), 0), view_55, out=buf91)
        del view_55
        buf92 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf89, buf92, 4704, 128, grid=grid(4704), stream=stream0)
        del buf89
        buf93 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf92, buf93, 196, 24, grid=grid(196), stream=stream0)
        buf94 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf90, buf94, 384, 1568, grid=grid(384), stream=stream0)
        buf95 = reinterpret_tensor(buf87, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf90, buf8, buf95, 4992, 121, grid=grid(4992), stream=stream0)
        buf96 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf95, buf96, 384, 13, grid=grid(384), stream=stream0)
        buf97 = buf70; del buf70  # reuse
        buf100 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf97, buf82, primals_60, buf90, primals_57, primals_52, buf100, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_52
        del primals_57
        del primals_60
        buf101 = reinterpret_tensor(buf80, (1568, 1536), (1536, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1568, 384), (384, 1), 0), permute_108, out=buf101)
        del permute_108
        buf107 = reinterpret_tensor(buf74, (8, 196, 1536), (301056, 1536, 1), 0); del buf74  # reuse
        # Source Nodes: [x_69, x_70], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf101, mm_8, primals_128, buf107, 2408448, grid=grid(2408448), stream=stream0)
        buf109 = reinterpret_tensor(buf90, (1568, 384), (384, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1568, 1536), (1536, 1), 0), permute_114, out=buf109)
        del permute_114
        buf98 = reinterpret_tensor(buf95, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf95  # reuse
        buf110 = buf85; del buf85  # reuse
        buf112 = buf83; del buf83  # reuse
        buf114 = reinterpret_tensor(buf76, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf76  # reuse
        # Source Nodes: [mul_16, x_68], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf97, addmm_17, buf109, buf7, primals_49, addmm_16, primals_54, buf98, buf110, buf112, buf114, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_16
        del addmm_17
        buf99 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf98, buf99, 384, 13, grid=grid(384), stream=stream0)
        buf102 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (384, 1568), (1, 384), 0), view_53, out=buf102)
        del view_53
        buf103 = reinterpret_tensor(buf98, (1, 384, 13), (4992, 1, 384), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf100, buf103, 4992, 121, grid=grid(4992), stream=stream0)
        buf104 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf103, buf104, 384, 13, grid=grid(384), stream=stream0)
        buf105 = buf78; del buf78  # reuse
        # Source Nodes: [x_69, x_70], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf101, mm_8, primals_128, buf105, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_8
        del primals_128
        buf106 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69, x_70], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf105, buf106, 1536, 13, grid=grid(1536), stream=stream0)
        buf108 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1536, 1568), (1, 1536), 0), view_51, out=buf108)
        del view_51
        buf111 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf110, buf111, 384, 13, grid=grid(384), stream=stream0)
        buf113 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_16, x_68], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf112, buf113, 384, 13, grid=grid(384), stream=stream0)
        buf115 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf114, buf115, 384, 13, grid=grid(384), stream=stream0)
        buf116 = reinterpret_tensor(buf100, (8, 384, 196), (75264, 196, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf97, buf109, primals_54, primals_49, buf116, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_49
        buf117 = reinterpret_tensor(buf82, (3072, 196), (196, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (3072, 196), (196, 1), 0), permute_117, out=buf117)
        del permute_117
        buf118 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (196, 3072), (1, 196), 0), view_49, out=buf118)
        del view_49
        buf119 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf116, buf119, 4704, 128, grid=grid(4704), stream=stream0)
        del buf116
        buf120 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf119, buf120, 196, 24, grid=grid(196), stream=stream0)
        buf121 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf117, buf121, 384, 1568, grid=grid(384), stream=stream0)
        buf122 = reinterpret_tensor(buf114, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf117, buf7, buf122, 4992, 121, grid=grid(4992), stream=stream0)
        buf123 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf122, buf123, 384, 13, grid=grid(384), stream=stream0)
        buf124 = reinterpret_tensor(buf109, (8, 196, 384), (75264, 384, 1), 0); del buf109  # reuse
        buf127 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf124, buf97, primals_54, buf117, primals_51, primals_46, buf127, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_46
        del primals_51
        del primals_54
        buf128 = reinterpret_tensor(buf107, (1568, 1536), (1536, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (1568, 384), (384, 1), 0), permute_122, out=buf128)
        del permute_122
        buf134 = reinterpret_tensor(buf101, (8, 196, 1536), (301056, 1536, 1), 0); del buf101  # reuse
        # Source Nodes: [x_61, x_62], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf128, mm_7, primals_122, buf134, 2408448, grid=grid(2408448), stream=stream0)
        buf136 = reinterpret_tensor(buf97, (1568, 384), (384, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (1568, 1536), (1536, 1), 0), permute_128, out=buf136)
        del permute_128
        buf125 = reinterpret_tensor(buf122, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf122  # reuse
        buf137 = buf112; del buf112  # reuse
        buf139 = buf110; del buf110  # reuse
        buf141 = reinterpret_tensor(buf103, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf103  # reuse
        # Source Nodes: [mul_14, x_60], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf124, addmm_15, buf136, buf6, primals_43, addmm_14, primals_48, buf125, buf137, buf139, buf141, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_14
        del addmm_15
        buf126 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf125, buf126, 384, 13, grid=grid(384), stream=stream0)
        buf129 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (384, 1568), (1, 384), 0), view_47, out=buf129)
        del view_47
        buf130 = reinterpret_tensor(buf125, (1, 384, 13), (4992, 1, 384), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf127, buf130, 4992, 121, grid=grid(4992), stream=stream0)
        buf131 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf130, buf131, 384, 13, grid=grid(384), stream=stream0)
        buf132 = buf105; del buf105  # reuse
        # Source Nodes: [x_61, x_62], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf128, mm_7, primals_122, buf132, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_7
        del primals_122
        buf133 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_62], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf132, buf133, 1536, 13, grid=grid(1536), stream=stream0)
        buf135 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (1536, 1568), (1, 1536), 0), view_45, out=buf135)
        del view_45
        buf138 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf137, buf138, 384, 13, grid=grid(384), stream=stream0)
        buf140 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_14, x_60], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf139, buf140, 384, 13, grid=grid(384), stream=stream0)
        buf142 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf141, buf142, 384, 13, grid=grid(384), stream=stream0)
        buf143 = reinterpret_tensor(buf127, (8, 384, 196), (75264, 196, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf124, buf136, primals_48, primals_43, buf143, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_43
        buf144 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (3072, 196), (196, 1), 0), permute_131, out=buf144)
        del permute_131
        buf145 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (196, 3072), (1, 196), 0), view_43, out=buf145)
        del view_43
        buf146 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf143, buf146, 4704, 128, grid=grid(4704), stream=stream0)
        del buf143
        buf147 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf146, buf147, 196, 24, grid=grid(196), stream=stream0)
        buf148 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf144, buf148, 384, 1568, grid=grid(384), stream=stream0)
        buf149 = reinterpret_tensor(buf141, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf144, buf6, buf149, 4992, 121, grid=grid(4992), stream=stream0)
        buf150 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf149, buf150, 384, 13, grid=grid(384), stream=stream0)
        buf151 = buf124; del buf124  # reuse
        buf154 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf151, buf136, primals_48, buf144, primals_45, primals_40, buf154, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_40
        del primals_45
        del primals_48
        buf155 = reinterpret_tensor(buf134, (1568, 1536), (1536, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1568, 384), (384, 1), 0), permute_136, out=buf155)
        del permute_136
        buf161 = reinterpret_tensor(buf128, (8, 196, 1536), (301056, 1536, 1), 0); del buf128  # reuse
        # Source Nodes: [x_53, x_54], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf155, mm_6, primals_116, buf161, 2408448, grid=grid(2408448), stream=stream0)
        buf163 = reinterpret_tensor(buf144, (1568, 384), (384, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (1568, 1536), (1536, 1), 0), permute_142, out=buf163)
        del permute_142
        buf152 = reinterpret_tensor(buf149, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf149  # reuse
        buf164 = buf139; del buf139  # reuse
        buf166 = buf137; del buf137  # reuse
        buf168 = reinterpret_tensor(buf130, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf130  # reuse
        # Source Nodes: [mul_12, x_52], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf151, addmm_13, buf163, buf5, primals_37, addmm_12, primals_42, buf152, buf164, buf166, buf168, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_12
        del addmm_13
        buf153 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf152, buf153, 384, 13, grid=grid(384), stream=stream0)
        buf156 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (384, 1568), (1, 384), 0), view_41, out=buf156)
        del view_41
        buf157 = reinterpret_tensor(buf152, (1, 384, 13), (4992, 1, 384), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf154, buf157, 4992, 121, grid=grid(4992), stream=stream0)
        buf158 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf157, buf158, 384, 13, grid=grid(384), stream=stream0)
        buf159 = buf132; del buf132  # reuse
        # Source Nodes: [x_53, x_54], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf155, mm_6, primals_116, buf159, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_6
        del primals_116
        buf160 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53, x_54], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf159, buf160, 1536, 13, grid=grid(1536), stream=stream0)
        buf162 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (1536, 1568), (1, 1536), 0), view_39, out=buf162)
        del view_39
        buf165 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf164, buf165, 384, 13, grid=grid(384), stream=stream0)
        buf167 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_12, x_52], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf166, buf167, 384, 13, grid=grid(384), stream=stream0)
        buf169 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf168, buf169, 384, 13, grid=grid(384), stream=stream0)
        buf170 = reinterpret_tensor(buf154, (8, 384, 196), (75264, 196, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf151, buf163, primals_42, primals_37, buf170, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_37
        buf171 = reinterpret_tensor(buf136, (3072, 196), (196, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (3072, 196), (196, 1), 0), permute_145, out=buf171)
        del permute_145
        buf172 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (196, 3072), (1, 196), 0), view_37, out=buf172)
        del view_37
        buf173 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf170, buf173, 4704, 128, grid=grid(4704), stream=stream0)
        del buf170
        buf174 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf173, buf174, 196, 24, grid=grid(196), stream=stream0)
        buf175 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf171, buf175, 384, 1568, grid=grid(384), stream=stream0)
        buf176 = reinterpret_tensor(buf168, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf171, buf5, buf176, 4992, 121, grid=grid(4992), stream=stream0)
        buf177 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf176, buf177, 384, 13, grid=grid(384), stream=stream0)
        buf178 = buf151; del buf151  # reuse
        buf181 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf178, buf163, primals_42, buf171, primals_39, primals_34, buf181, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_34
        del primals_39
        del primals_42
        buf182 = reinterpret_tensor(buf161, (1568, 1536), (1536, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (1568, 384), (384, 1), 0), permute_150, out=buf182)
        del permute_150
        buf188 = reinterpret_tensor(buf155, (8, 196, 1536), (301056, 1536, 1), 0); del buf155  # reuse
        # Source Nodes: [x_45, x_46], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf182, mm_5, primals_110, buf188, 2408448, grid=grid(2408448), stream=stream0)
        buf190 = reinterpret_tensor(buf171, (1568, 384), (384, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1568, 1536), (1536, 1), 0), permute_156, out=buf190)
        del permute_156
        buf179 = reinterpret_tensor(buf176, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf176  # reuse
        buf191 = buf166; del buf166  # reuse
        buf193 = buf164; del buf164  # reuse
        buf195 = reinterpret_tensor(buf157, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf157  # reuse
        # Source Nodes: [mul_10, x_44], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf178, addmm_11, buf190, buf4, primals_31, addmm_10, primals_36, buf179, buf191, buf193, buf195, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_10
        del addmm_11
        buf180 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf179, buf180, 384, 13, grid=grid(384), stream=stream0)
        buf183 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (384, 1568), (1, 384), 0), view_35, out=buf183)
        del view_35
        buf184 = reinterpret_tensor(buf179, (1, 384, 13), (4992, 1, 384), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf181, buf184, 4992, 121, grid=grid(4992), stream=stream0)
        buf185 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf184, buf185, 384, 13, grid=grid(384), stream=stream0)
        buf186 = buf159; del buf159  # reuse
        # Source Nodes: [x_45, x_46], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf182, mm_5, primals_110, buf186, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_5
        del primals_110
        buf187 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_46], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf186, buf187, 1536, 13, grid=grid(1536), stream=stream0)
        buf189 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1536, 1568), (1, 1536), 0), view_33, out=buf189)
        del view_33
        buf192 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf191, buf192, 384, 13, grid=grid(384), stream=stream0)
        buf194 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, x_44], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf193, buf194, 384, 13, grid=grid(384), stream=stream0)
        buf196 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf195, buf196, 384, 13, grid=grid(384), stream=stream0)
        buf197 = reinterpret_tensor(buf181, (8, 384, 196), (75264, 196, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf178, buf190, primals_36, primals_31, buf197, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_31
        buf198 = reinterpret_tensor(buf163, (3072, 196), (196, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (3072, 196), (196, 1), 0), permute_159, out=buf198)
        del permute_159
        buf199 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (196, 3072), (1, 196), 0), view_31, out=buf199)
        del view_31
        buf200 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf197, buf200, 4704, 128, grid=grid(4704), stream=stream0)
        del buf197
        buf201 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf200, buf201, 196, 24, grid=grid(196), stream=stream0)
        buf202 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf198, buf202, 384, 1568, grid=grid(384), stream=stream0)
        buf203 = reinterpret_tensor(buf195, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf198, buf4, buf203, 4992, 121, grid=grid(4992), stream=stream0)
        buf204 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf203, buf204, 384, 13, grid=grid(384), stream=stream0)
        buf205 = buf178; del buf178  # reuse
        buf208 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf205, buf190, primals_36, buf198, primals_33, primals_28, buf208, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_28
        del primals_33
        del primals_36
        buf209 = reinterpret_tensor(buf188, (1568, 1536), (1536, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1568, 384), (384, 1), 0), permute_164, out=buf209)
        del permute_164
        buf215 = reinterpret_tensor(buf182, (8, 196, 1536), (301056, 1536, 1), 0); del buf182  # reuse
        # Source Nodes: [x_37, x_38], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf209, mm_4, primals_104, buf215, 2408448, grid=grid(2408448), stream=stream0)
        buf217 = reinterpret_tensor(buf198, (1568, 384), (384, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (1568, 1536), (1536, 1), 0), permute_170, out=buf217)
        del permute_170
        buf206 = reinterpret_tensor(buf203, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf203  # reuse
        buf218 = buf193; del buf193  # reuse
        buf220 = buf191; del buf191  # reuse
        buf222 = reinterpret_tensor(buf184, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf184  # reuse
        # Source Nodes: [mul_8, x_36], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf205, addmm_9, buf217, buf3, primals_25, addmm_8, primals_30, buf206, buf218, buf220, buf222, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_8
        del addmm_9
        buf207 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf206, buf207, 384, 13, grid=grid(384), stream=stream0)
        buf210 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (384, 1568), (1, 384), 0), view_29, out=buf210)
        del view_29
        buf211 = reinterpret_tensor(buf206, (1, 384, 13), (4992, 1, 384), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf208, buf211, 4992, 121, grid=grid(4992), stream=stream0)
        buf212 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf211, buf212, 384, 13, grid=grid(384), stream=stream0)
        buf213 = buf186; del buf186  # reuse
        # Source Nodes: [x_37, x_38], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf209, mm_4, primals_104, buf213, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_4
        del primals_104
        buf214 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37, x_38], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf213, buf214, 1536, 13, grid=grid(1536), stream=stream0)
        buf216 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (1536, 1568), (1, 1536), 0), view_27, out=buf216)
        del view_27
        buf219 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf218, buf219, 384, 13, grid=grid(384), stream=stream0)
        buf221 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_8, x_36], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf220, buf221, 384, 13, grid=grid(384), stream=stream0)
        buf223 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf222, buf223, 384, 13, grid=grid(384), stream=stream0)
        buf224 = reinterpret_tensor(buf208, (8, 384, 196), (75264, 196, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf205, buf217, primals_30, primals_25, buf224, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_25
        buf225 = reinterpret_tensor(buf190, (3072, 196), (196, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (3072, 196), (196, 1), 0), permute_173, out=buf225)
        del permute_173
        buf226 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (196, 3072), (1, 196), 0), view_25, out=buf226)
        del view_25
        buf227 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf224, buf227, 4704, 128, grid=grid(4704), stream=stream0)
        del buf224
        buf228 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf227, buf228, 196, 24, grid=grid(196), stream=stream0)
        buf229 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf225, buf229, 384, 1568, grid=grid(384), stream=stream0)
        buf230 = reinterpret_tensor(buf222, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf225, buf3, buf230, 4992, 121, grid=grid(4992), stream=stream0)
        buf231 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf230, buf231, 384, 13, grid=grid(384), stream=stream0)
        buf232 = buf205; del buf205  # reuse
        buf235 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf232, buf217, primals_30, buf225, primals_27, primals_22, buf235, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_22
        del primals_27
        del primals_30
        buf236 = reinterpret_tensor(buf215, (1568, 1536), (1536, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (1568, 384), (384, 1), 0), permute_178, out=buf236)
        del permute_178
        buf242 = reinterpret_tensor(buf209, (8, 196, 1536), (301056, 1536, 1), 0); del buf209  # reuse
        # Source Nodes: [x_29, x_30], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf236, mm_3, primals_98, buf242, 2408448, grid=grid(2408448), stream=stream0)
        buf244 = reinterpret_tensor(buf225, (1568, 384), (384, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 1536), (1536, 1), 0), permute_184, out=buf244)
        del permute_184
        buf233 = reinterpret_tensor(buf230, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf230  # reuse
        buf245 = buf220; del buf220  # reuse
        buf247 = buf218; del buf218  # reuse
        buf249 = reinterpret_tensor(buf211, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf211  # reuse
        # Source Nodes: [mul_6, x_28], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf232, addmm_7, buf244, buf2, primals_19, addmm_6, primals_24, buf233, buf245, buf247, buf249, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_6
        del addmm_7
        buf234 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf233, buf234, 384, 13, grid=grid(384), stream=stream0)
        buf237 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (384, 1568), (1, 384), 0), view_23, out=buf237)
        del view_23
        buf238 = reinterpret_tensor(buf233, (1, 384, 13), (4992, 1, 384), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf235, buf238, 4992, 121, grid=grid(4992), stream=stream0)
        buf239 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf238, buf239, 384, 13, grid=grid(384), stream=stream0)
        buf240 = buf213; del buf213  # reuse
        # Source Nodes: [x_29, x_30], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf236, mm_3, primals_98, buf240, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_3
        del primals_98
        buf241 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_30], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf240, buf241, 1536, 13, grid=grid(1536), stream=stream0)
        buf243 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1536, 1568), (1, 1536), 0), view_21, out=buf243)
        del view_21
        buf246 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf245, buf246, 384, 13, grid=grid(384), stream=stream0)
        buf248 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_6, x_28], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf247, buf248, 384, 13, grid=grid(384), stream=stream0)
        buf250 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf249, buf250, 384, 13, grid=grid(384), stream=stream0)
        buf251 = reinterpret_tensor(buf235, (8, 384, 196), (75264, 196, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf232, buf244, primals_24, primals_19, buf251, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_19
        buf252 = reinterpret_tensor(buf217, (3072, 196), (196, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (3072, 196), (196, 1), 0), permute_187, out=buf252)
        del permute_187
        buf253 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (196, 3072), (1, 196), 0), view_19, out=buf253)
        del view_19
        buf254 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf251, buf254, 4704, 128, grid=grid(4704), stream=stream0)
        del buf251
        buf255 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf254, buf255, 196, 24, grid=grid(196), stream=stream0)
        buf256 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf252, buf256, 384, 1568, grid=grid(384), stream=stream0)
        buf257 = reinterpret_tensor(buf249, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf252, buf2, buf257, 4992, 121, grid=grid(4992), stream=stream0)
        buf258 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf257, buf258, 384, 13, grid=grid(384), stream=stream0)
        buf259 = buf232; del buf232  # reuse
        buf262 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf259, buf244, primals_24, buf252, primals_21, primals_16, buf262, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_16
        del primals_21
        del primals_24
        buf263 = reinterpret_tensor(buf242, (1568, 1536), (1536, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1568, 384), (384, 1), 0), permute_192, out=buf263)
        del permute_192
        buf269 = reinterpret_tensor(buf236, (8, 196, 1536), (301056, 1536, 1), 0); del buf236  # reuse
        # Source Nodes: [x_21, x_22], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf263, mm_2, primals_92, buf269, 2408448, grid=grid(2408448), stream=stream0)
        buf271 = reinterpret_tensor(buf252, (1568, 384), (384, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (1568, 1536), (1536, 1), 0), permute_198, out=buf271)
        del permute_198
        buf260 = reinterpret_tensor(buf257, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf257  # reuse
        buf272 = buf247; del buf247  # reuse
        buf274 = buf245; del buf245  # reuse
        buf276 = reinterpret_tensor(buf238, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf238  # reuse
        # Source Nodes: [mul_4, x_20], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf259, addmm_5, buf271, buf1, primals_13, addmm_4, primals_18, buf260, buf272, buf274, buf276, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_4
        del addmm_5
        buf261 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf260, buf261, 384, 13, grid=grid(384), stream=stream0)
        buf264 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (384, 1568), (1, 384), 0), view_17, out=buf264)
        del view_17
        buf265 = reinterpret_tensor(buf260, (1, 384, 13), (4992, 1, 384), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf262, buf265, 4992, 121, grid=grid(4992), stream=stream0)
        buf266 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf265, buf266, 384, 13, grid=grid(384), stream=stream0)
        del buf265
        buf267 = buf240; del buf240  # reuse
        # Source Nodes: [x_21, x_22], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf263, mm_2, primals_92, buf267, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_2
        del primals_92
        buf268 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_22], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf267, buf268, 1536, 13, grid=grid(1536), stream=stream0)
        buf270 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (1536, 1568), (1, 1536), 0), view_15, out=buf270)
        del view_15
        buf273 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf272, buf273, 384, 13, grid=grid(384), stream=stream0)
        buf275 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_4, x_20], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf274, buf275, 384, 13, grid=grid(384), stream=stream0)
        buf277 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf276, buf277, 384, 13, grid=grid(384), stream=stream0)
        buf278 = reinterpret_tensor(buf262, (8, 384, 196), (75264, 196, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf259, buf271, primals_18, primals_13, buf278, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_13
        buf279 = reinterpret_tensor(buf244, (3072, 196), (196, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (3072, 196), (196, 1), 0), permute_201, out=buf279)
        del permute_201
        buf280 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (196, 3072), (1, 196), 0), view_13, out=buf280)
        del view_13
        buf281 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf278, buf281, 4704, 128, grid=grid(4704), stream=stream0)
        del buf278
        buf282 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf281, buf282, 196, 24, grid=grid(196), stream=stream0)
        buf283 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf279, buf283, 384, 1568, grid=grid(384), stream=stream0)
        buf284 = reinterpret_tensor(buf276, (1, 1, 384, 13), (4992, 4992, 13, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_17.run(buf279, buf1, buf284, 4992, 121, grid=grid(4992), stream=stream0)
        buf285 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf284, buf285, 384, 13, grid=grid(384), stream=stream0)
        buf286 = buf259; del buf259  # reuse
        buf289 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf286, buf271, primals_18, buf279, primals_15, primals_10, buf289, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_10
        del primals_15
        del primals_18
        buf290 = reinterpret_tensor(buf269, (1568, 1536), (1536, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1568, 384), (384, 1), 0), permute_206, out=buf290)
        del permute_206
        buf296 = reinterpret_tensor(buf263, (8, 196, 1536), (301056, 1536, 1), 0); del buf263  # reuse
        # Source Nodes: [x_13, x_14], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf290, mm_1, primals_86, buf296, 2408448, grid=grid(2408448), stream=stream0)
        buf298 = reinterpret_tensor(buf279, (1568, 384), (384, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (1568, 1536), (1536, 1), 0), permute_212, out=buf298)
        del permute_212
        buf287 = reinterpret_tensor(buf284, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf284  # reuse
        buf299 = buf274; del buf274  # reuse
        buf302 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_22.run(buf286, addmm_3, buf298, primals_12, addmm_2, buf287, buf299, buf302, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_3
        buf288 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf287, buf288, 384, 13, grid=grid(384), stream=stream0)
        buf291 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (384, 1568), (1, 384), 0), view_11, out=buf291)
        del view_11
        buf292 = reinterpret_tensor(buf287, (1, 384, 13), (4992, 1, 384), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf289, buf292, 4992, 121, grid=grid(4992), stream=stream0)
        buf293 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf292, buf293, 384, 13, grid=grid(384), stream=stream0)
        buf294 = buf267; del buf267  # reuse
        # Source Nodes: [x_13, x_14], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf290, mm_1, primals_86, buf294, 19968, 121, grid=grid(19968), stream=stream0)
        del mm_1
        del primals_86
        buf295 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_14], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf294, buf295, 1536, 13, grid=grid(1536), stream=stream0)
        buf297 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (1536, 1568), (1, 1536), 0), view_9, out=buf297)
        del view_9
        buf300 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf299, buf300, 384, 13, grid=grid(384), stream=stream0)
        buf304 = reinterpret_tensor(buf289, (8, 384, 196), (75264, 196, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf286, buf298, primals_12, primals_7, buf304, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf305 = reinterpret_tensor(buf271, (3072, 196), (196, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (3072, 196), (196, 1), 0), permute_215, out=buf305)
        del permute_215
        buf301 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        buf309 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        buf310 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_2, x_12], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf298, buf0, primals_7, addmm_2, buf305, buf301, buf309, buf310, 384, 1568, grid=grid(384), stream=stream0)
        del addmm_2
        del buf0
        del primals_7
        buf303 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf302, buf303, 384, 13, grid=grid(384), stream=stream0)
        buf306 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (196, 3072), (1, 196), 0), view_7, out=buf306)
        del view_7
        buf307 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf304, buf307, 4704, 128, grid=grid(4704), stream=stream0)
        buf308 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf307, buf308, 196, 24, grid=grid(196), stream=stream0)
        buf311 = buf286; del buf286  # reuse
        buf314 = reinterpret_tensor(buf304, (8, 196, 384), (75264, 384, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf311, buf298, primals_12, buf305, primals_9, primals_4, buf314, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_12
        del primals_4
        del primals_9
        buf315 = reinterpret_tensor(buf296, (1568, 1536), (1536, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (1568, 384), (384, 1), 0), permute_220, out=buf315)
        del permute_220
        buf321 = reinterpret_tensor(buf290, (8, 196, 1536), (301056, 1536, 1), 0); del buf290  # reuse
        # Source Nodes: [x_5, x_6], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_2.run(buf315, mm, primals_80, buf321, 2408448, grid=grid(2408448), stream=stream0)
        buf323 = reinterpret_tensor(buf305, (1568, 384), (384, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1568, 1536), (1536, 1), 0), permute_226, out=buf323)
        del permute_226
        buf312 = buf302; del buf302  # reuse
        buf324 = buf299; del buf299  # reuse
        buf327 = reinterpret_tensor(buf292, (1, 1, 384, 13), (4992, 4992, 1, 384), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_22.run(buf311, addmm_1, buf323, primals_6, addmm, buf312, buf324, buf327, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_1
        buf313 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf312, buf313, 384, 13, grid=grid(384), stream=stream0)
        buf316 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (384, 1568), (1, 384), 0), view_5, out=buf316)
        del view_5
        buf317 = reinterpret_tensor(buf312, (1, 384, 13), (4992, 1, 384), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf314, buf317, 4992, 121, grid=grid(4992), stream=stream0)
        buf318 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf317, buf318, 384, 13, grid=grid(384), stream=stream0)
        del buf317
        buf319 = buf294; del buf294  # reuse
        # Source Nodes: [x_5, x_6], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_10.run(buf315, mm, primals_80, buf319, 19968, 121, grid=grid(19968), stream=stream0)
        del buf315
        del mm
        del primals_80
        buf320 = empty((1, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_6], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_11.run(buf319, buf320, 1536, 13, grid=grid(1536), stream=stream0)
        del buf319
        buf322 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1536, 1568), (1, 1536), 0), view_3, out=buf322)
        del buf321
        del view_3
        buf325 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf324, buf325, 384, 13, grid=grid(384), stream=stream0)
        del buf324
        buf329 = reinterpret_tensor(buf314, (8, 384, 196), (75264, 196, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf311, buf323, primals_6, primals_1, buf329, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf330 = reinterpret_tensor(buf298, (3072, 196), (196, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (3072, 196), (196, 1), 0), permute_229, out=buf330)
        del permute_229
        buf326 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        buf334 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        buf335 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, x_4], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf323, convolution, primals_1, addmm, buf330, buf326, buf334, buf335, 384, 1568, grid=grid(384), stream=stream0)
        del addmm
        del convolution
        del primals_1
        buf328 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_8.run(buf327, buf328, 384, 13, grid=grid(384), stream=stream0)
        buf331 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (196, 3072), (1, 196), 0), view_1, out=buf331)
        del view_1
        buf332 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf329, buf332, 4704, 128, grid=grid(4704), stream=stream0)
        del buf329
        buf333 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf332, buf333, 196, 24, grid=grid(196), stream=stream0)
        del buf332
        buf336 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_24.run(buf336, buf323, primals_6, buf330, primals_3, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf323
        del buf330
        del primals_3
        del primals_6
        buf337 = reinterpret_tensor(buf327, (384, 13), (1, 384), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_25.run(buf336, buf337, 4992, 121, grid=grid(4992), stream=stream0)
        buf338 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_add_div_mul_sum_8.run(buf337, buf338, 384, 13, grid=grid(384), stream=stream0)
        del buf337
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf339 = aten.convolution_backward(reinterpret_tensor(buf336, (8, 384, 14, 14), (75264, 1, 5376, 384), 0), primals_151, primals_75, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf336
        del primals_151
        del primals_75
        buf340 = buf339[1]
        return (reinterpret_tensor(buf328, (384, ), (1, ), 0), buf334, buf335, reinterpret_tensor(buf313, (384, ), (1, ), 0), buf325, buf326, reinterpret_tensor(buf303, (384, ), (1, ), 0), buf309, buf310, reinterpret_tensor(buf288, (384, ), (1, ), 0), buf300, buf301, reinterpret_tensor(buf277, (384, ), (1, ), 0), buf283, buf285, reinterpret_tensor(buf261, (384, ), (1, ), 0), buf273, buf275, reinterpret_tensor(buf250, (384, ), (1, ), 0), buf256, buf258, reinterpret_tensor(buf234, (384, ), (1, ), 0), buf246, buf248, reinterpret_tensor(buf223, (384, ), (1, ), 0), buf229, buf231, reinterpret_tensor(buf207, (384, ), (1, ), 0), buf219, buf221, reinterpret_tensor(buf196, (384, ), (1, ), 0), buf202, buf204, reinterpret_tensor(buf180, (384, ), (1, ), 0), buf192, buf194, reinterpret_tensor(buf169, (384, ), (1, ), 0), buf175, buf177, reinterpret_tensor(buf153, (384, ), (1, ), 0), buf165, buf167, reinterpret_tensor(buf142, (384, ), (1, ), 0), buf148, buf150, reinterpret_tensor(buf126, (384, ), (1, ), 0), buf138, buf140, reinterpret_tensor(buf115, (384, ), (1, ), 0), buf121, buf123, reinterpret_tensor(buf99, (384, ), (1, ), 0), buf111, buf113, reinterpret_tensor(buf88, (384, ), (1, ), 0), buf94, buf96, reinterpret_tensor(buf72, (384, ), (1, ), 0), buf84, buf86, reinterpret_tensor(buf61, (384, ), (1, ), 0), buf67, buf69, reinterpret_tensor(buf45, (384, ), (1, ), 0), buf57, buf59, reinterpret_tensor(buf34, (384, ), (1, ), 0), buf40, buf42, reinterpret_tensor(buf18, (384, ), (1, ), 0), buf30, buf32, buf14, buf16, buf340, buf338, reinterpret_tensor(buf331, (196, 196), (196, 1), 0), reinterpret_tensor(buf333, (196, ), (1, ), 0), reinterpret_tensor(buf322, (1536, 384), (384, 1), 0), reinterpret_tensor(buf320, (1536, ), (1, ), 0), reinterpret_tensor(buf316, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf318, (384, ), (1, ), 0), reinterpret_tensor(buf306, (196, 196), (196, 1), 0), reinterpret_tensor(buf308, (196, ), (1, ), 0), reinterpret_tensor(buf297, (1536, 384), (384, 1), 0), reinterpret_tensor(buf295, (1536, ), (1, ), 0), reinterpret_tensor(buf291, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf293, (384, ), (1, ), 0), reinterpret_tensor(buf280, (196, 196), (196, 1), 0), reinterpret_tensor(buf282, (196, ), (1, ), 0), reinterpret_tensor(buf270, (1536, 384), (384, 1), 0), reinterpret_tensor(buf268, (1536, ), (1, ), 0), reinterpret_tensor(buf264, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf266, (384, ), (1, ), 0), reinterpret_tensor(buf253, (196, 196), (196, 1), 0), reinterpret_tensor(buf255, (196, ), (1, ), 0), reinterpret_tensor(buf243, (1536, 384), (384, 1), 0), reinterpret_tensor(buf241, (1536, ), (1, ), 0), reinterpret_tensor(buf237, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf239, (384, ), (1, ), 0), reinterpret_tensor(buf226, (196, 196), (196, 1), 0), reinterpret_tensor(buf228, (196, ), (1, ), 0), reinterpret_tensor(buf216, (1536, 384), (384, 1), 0), reinterpret_tensor(buf214, (1536, ), (1, ), 0), reinterpret_tensor(buf210, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf212, (384, ), (1, ), 0), reinterpret_tensor(buf199, (196, 196), (196, 1), 0), reinterpret_tensor(buf201, (196, ), (1, ), 0), reinterpret_tensor(buf189, (1536, 384), (384, 1), 0), reinterpret_tensor(buf187, (1536, ), (1, ), 0), reinterpret_tensor(buf183, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf185, (384, ), (1, ), 0), reinterpret_tensor(buf172, (196, 196), (196, 1), 0), reinterpret_tensor(buf174, (196, ), (1, ), 0), reinterpret_tensor(buf162, (1536, 384), (384, 1), 0), reinterpret_tensor(buf160, (1536, ), (1, ), 0), reinterpret_tensor(buf156, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf158, (384, ), (1, ), 0), reinterpret_tensor(buf145, (196, 196), (196, 1), 0), reinterpret_tensor(buf147, (196, ), (1, ), 0), reinterpret_tensor(buf135, (1536, 384), (384, 1), 0), reinterpret_tensor(buf133, (1536, ), (1, ), 0), reinterpret_tensor(buf129, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf131, (384, ), (1, ), 0), reinterpret_tensor(buf118, (196, 196), (196, 1), 0), reinterpret_tensor(buf120, (196, ), (1, ), 0), reinterpret_tensor(buf108, (1536, 384), (384, 1), 0), reinterpret_tensor(buf106, (1536, ), (1, ), 0), reinterpret_tensor(buf102, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf104, (384, ), (1, ), 0), reinterpret_tensor(buf91, (196, 196), (196, 1), 0), reinterpret_tensor(buf93, (196, ), (1, ), 0), reinterpret_tensor(buf81, (1536, 384), (384, 1), 0), reinterpret_tensor(buf79, (1536, ), (1, ), 0), reinterpret_tensor(buf75, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf77, (384, ), (1, ), 0), reinterpret_tensor(buf64, (196, 196), (196, 1), 0), reinterpret_tensor(buf66, (196, ), (1, ), 0), reinterpret_tensor(buf54, (1536, 384), (384, 1), 0), reinterpret_tensor(buf52, (1536, ), (1, ), 0), reinterpret_tensor(buf48, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf50, (384, ), (1, ), 0), reinterpret_tensor(buf37, (196, 196), (196, 1), 0), reinterpret_tensor(buf39, (196, ), (1, ), 0), reinterpret_tensor(buf27, (1536, 384), (384, 1), 0), reinterpret_tensor(buf25, (1536, ), (1, ), 0), reinterpret_tensor(buf21, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf23, (384, ), (1, ), 0), reinterpret_tensor(buf12, (1000, 384), (384, 1), 0), reinterpret_tensor(buf13, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_9 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_15 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_10 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    clone_36 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_62 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_89 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_215 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, convolution, view_1, addmm, view_3, mm, view_5, addmm_1, view_7, addmm_2, view_9, mm_1, view_11, addmm_3, view_13, addmm_4, view_15, mm_2, view_17, addmm_5, view_19, addmm_6, view_21, mm_3, view_23, addmm_7, view_25, addmm_8, view_27, mm_4, view_29, addmm_9, view_31, addmm_10, view_33, mm_5, view_35, addmm_11, view_37, addmm_12, view_39, mm_6, view_41, addmm_13, view_43, addmm_14, view_45, mm_7, view_47, addmm_15, view_49, addmm_16, view_51, mm_8, view_53, addmm_17, view_55, addmm_18, view_57, mm_9, view_59, addmm_19, view_61, addmm_20, view_63, mm_10, view_65, addmm_21, view_67, addmm_22, view_69, mm_11, view_71, addmm_23, clone_36, permute_62, permute_66, permute_72, permute_75, permute_80, permute_86, permute_89, permute_94, permute_100, permute_103, permute_108, permute_114, permute_117, permute_122, permute_128, permute_131, permute_136, permute_142, permute_145, permute_150, permute_156, permute_159, permute_164, permute_170, permute_173, permute_178, permute_184, permute_187, permute_192, permute_198, permute_201, permute_206, permute_212, permute_215, permute_220, permute_226, permute_229, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
