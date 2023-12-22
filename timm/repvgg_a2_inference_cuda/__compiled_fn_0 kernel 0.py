
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxish6zmx4dui6pveazhbu4mcmarkjnsxp62dxxxaap5mpfurl7.py
# Source Nodes: [x, x_5], Original ATen: [aten.convolution]
# x => convolution
# x_5 => convolution_1
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5b76sdupxyn7nezhz226yrtzhzrbav3ccwxevg2lotiu6crky7.py
# Source Nodes: [x_5], Original ATen: [aten.convolution]
# x_5 => convolution_1
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/no/cnozsbxvjb6j6a6ywfjksuauktqj5gj7ygmjpa5juk6ybzel4b5c.py
# Source Nodes: [x_1, x_10, x_12, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_10 => add_4
# x_12 => relu
# x_6 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hq/chq4g56l3d6zmwhoffvdfeycxv3felk6dsxfoepu2qn76zjob6nb.py
# Source Nodes: [x_18], Original ATen: [aten.convolution]
# x_18 => convolution_3
triton_poi_fused_convolution_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/p6/cp65biig2dlsuxdrq6xroet66w2tzsxcgrtj3qsi5l5hrlaqyloh.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_14, x_19, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___0_____0___act => relu_1
# x_14 => add_6, mul_7, mul_8, sub_2
# x_19 => add_8, mul_10, mul_11, sub_3
# x_23 => add_9
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijnes4q5fnxop3z7qmuo33cgecx5p5tfoildndjc27vetnazqc5.py
# Source Nodes: [x_33], Original ATen: [aten.convolution]
# x_33 => convolution_5
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ym/cymfyglcojnzkuyzvf2sqqz5mzera7mzatinmhxqijrnxsns3zj2.py
# Source Nodes: [x_29, x_34, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_29 => add_13, mul_16, mul_17, sub_5
# x_34 => add_15, mul_19, mul_20, sub_6
# x_38 => add_16
triton_poi_fused__native_batch_norm_legit_no_training_add_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/coumzmjubgwbixgryioqezqoacxa4ae7ckhslohqivdoklxwox2t.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____1___act, x_25, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___0_____1___act => relu_2
# x_25 => add_11, mul_13, mul_14, sub_4
# x_40 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (96*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/csht2mjmdfn7uoukhlz72gxz6l7xxdfuockrnuctu4umzvbx6r6o.py
# Source Nodes: [x_47], Original ATen: [aten.convolution]
# x_47 => convolution_7
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dn/cdn3lsrhix47km66dghddw2fnssl3omghibyhoug2aurm2dev7oe.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_43, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___1_____0___act => relu_3
# x_43 => add_19, mul_22, mul_23, sub_7
# x_48 => add_21, mul_25, mul_26, sub_8
# x_52 => add_22
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chogqxj572hci76omtcuoge5kbwyhpqoru552dr72hoz6fc74wgn.py
# Source Nodes: [x_62], Original ATen: [aten.convolution]
# x_62 => convolution_9
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm2c6f2kdeqsh54r2j6vur6xox45jyfvfjwvt7tkhzwkb4ch73i.py
# Source Nodes: [x_58, x_63, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_58 => add_26, mul_31, mul_32, sub_10
# x_63 => add_28, mul_34, mul_35, sub_11
# x_67 => add_29
triton_poi_fused__native_batch_norm_legit_no_training_add_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmkrfw3ca6rkkikgxqxhalfhbtn34h27m234wbtnqg6hhdw7xgw.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act, x_54, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___1_____1___act => relu_4
# x_54 => add_24, mul_28, mul_29, sub_9
# x_69 => add_30
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidm3arrn5fpaq2v54wsornlczrx7rnstkpmya5a6gcns6rf5sc7.py
# Source Nodes: [x_110], Original ATen: [aten.convolution]
# x_110 => convolution_15
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ld/cldguzifp3ffnfvkkpkmfrtwfxtwr4w75mwdwkvu5n3bubq6r2zz.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_106, x_111, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___2_____0___act => relu_7
# x_106 => add_48, mul_55, mul_56, sub_18
# x_111 => add_50, mul_58, mul_59, sub_19
# x_115 => add_51
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3ql7fel2xdnm7pc4sy4auzqbz6uscpfw2w23bt4h4kntcovi4y.py
# Source Nodes: [x_125], Original ATen: [aten.convolution]
# x_125 => convolution_17
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4s/c4shrkcsytfmj6bii26pqost5rktzj4ia5pighslqubwiehjnsbh.py
# Source Nodes: [x_121, x_126, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_121 => add_55, mul_64, mul_65, sub_21
# x_126 => add_57, mul_67, mul_68, sub_22
# x_130 => add_58
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2i3kqti3jkcifzrkfryba25cluxj3djqlq4ib4lzmamaqgiouu.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act, x_117, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# getattr_getattr_l__mod___stages___2_____1___act => relu_8
# x_117 => add_53, mul_61, mul_62, sub_20
# x_132 => add_59
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffz4fbjczd4ctja2bepfhjdswj3tfeedll35vk7ysfla7einfsq.py
# Source Nodes: [x_343], Original ATen: [aten.convolution]
# x_343 => convolution_43
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7vdtjf5slgz6ki47if5hd75qbjjjsflminuvk4tyw34t2ttz4r.py
# Source Nodes: [x_339, x_344, x_348, x_350, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# x_339 => add_157, mul_178, mul_179, sub_59
# x_344 => add_159, mul_181, mul_182, sub_60
# x_348 => add_160
# x_350 => relu_21
# x_353 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1408
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 49.0
    tmp35 = tmp33 / tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp35, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, ), (1, ))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (96, ), (1, ))
    assert_size_stride(arg5_1, (96, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (96, ), (1, ))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (192, ), (1, ))
    assert_size_stride(arg18_1, (192, ), (1, ))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (192, ), (1, ))
    assert_size_stride(arg23_1, (192, ), (1, ))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (384, ), (1, ))
    assert_size_stride(arg37_1, (384, ), (1, ))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (384, ), (1, ))
    assert_size_stride(arg47_1, (384, ), (1, ))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (384, ), (1, ))
    assert_size_stride(arg59_1, (384, ), (1, ))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (384, ), (1, ))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (384, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (1408, ), (1, ))
    assert_size_stride(arg119_1, (1408, ), (1, ))
    assert_size_stride(arg120_1, (1408, ), (1, ))
    assert_size_stride(arg121_1, (1408, ), (1, ))
    assert_size_stride(arg122_1, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(arg123_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg124_1, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg125_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg126_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg127_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg128_1, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg129_1, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg130_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg131_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg132_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg133_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg134_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg135_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg136_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg137_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg138_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg139_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg140_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg141_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg142_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg143_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg144_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg145_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg146_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg147_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg148_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg149_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg150_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg151_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg152_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg153_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg154_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg155_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg156_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg157_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg158_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg159_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg160_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg161_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg162_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg163_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg164_1, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg165_1, (1408, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg166_1, (1000, 1408), (1408, 1))
    assert_size_stride(arg167_1, (1000, ), (1, ))
    assert_size_stride(arg168_1, (64, ), (1, ))
    assert_size_stride(arg169_1, (64, ), (1, ))
    assert_size_stride(arg170_1, (64, ), (1, ))
    assert_size_stride(arg171_1, (64, ), (1, ))
    assert_size_stride(arg172_1, (96, ), (1, ))
    assert_size_stride(arg173_1, (96, ), (1, ))
    assert_size_stride(arg174_1, (96, ), (1, ))
    assert_size_stride(arg175_1, (96, ), (1, ))
    assert_size_stride(arg176_1, (96, ), (1, ))
    assert_size_stride(arg177_1, (96, ), (1, ))
    assert_size_stride(arg178_1, (96, ), (1, ))
    assert_size_stride(arg179_1, (96, ), (1, ))
    assert_size_stride(arg180_1, (96, ), (1, ))
    assert_size_stride(arg181_1, (96, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, ), (1, ))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, ), (1, ))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (192, ), (1, ))
    assert_size_stride(arg192_1, (192, ), (1, ))
    assert_size_stride(arg193_1, (192, ), (1, ))
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, ), (1, ))
    assert_size_stride(arg196_1, (192, ), (1, ))
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (192, ), (1, ))
    assert_size_stride(arg199_1, (192, ), (1, ))
    assert_size_stride(arg200_1, (192, ), (1, ))
    assert_size_stride(arg201_1, (192, ), (1, ))
    assert_size_stride(arg202_1, (192, ), (1, ))
    assert_size_stride(arg203_1, (192, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (384, ), (1, ))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, ), (1, ))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (384, ), (1, ))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (384, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (384, ), (1, ))
    assert_size_stride(arg264_1, (384, ), (1, ))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, ), (1, ))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (384, ), (1, ))
    assert_size_stride(arg276_1, (384, ), (1, ))
    assert_size_stride(arg277_1, (384, ), (1, ))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (384, ), (1, ))
    assert_size_stride(arg283_1, (384, ), (1, ))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (1408, ), (1, ))
    assert_size_stride(arg287_1, (1408, ), (1, ))
    assert_size_stride(arg288_1, (1408, ), (1, ))
    assert_size_stride(arg289_1, (1408, ), (1, ))
    assert_size_stride(arg290_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x, x_5], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg290_1, buf0, buf2, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg290_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg122_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg122_1
        del buf0
        buf3 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg123_1, buf3, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg123_1
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del buf3
        buf5 = buf1; del buf1  # reuse
        buf6 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_10, x_12, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf5, arg168_1, arg169_1, arg0_1, arg1_1, buf4, arg170_1, arg171_1, arg2_1, arg3_1, buf6, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del arg0_1
        del arg168_1
        del arg169_1
        del arg170_1
        del arg171_1
        del arg1_1
        del arg2_1
        del arg3_1
        del buf4
        del buf5
        # Source Nodes: [x_12, x_13], Original ATen: [aten.convolution, aten.relu]
        buf7 = extern_kernels.convolution(buf6, arg124_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg124_1
        buf8 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg125_1, buf8, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg125_1
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf6, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del buf6
        del buf8
        buf10 = buf7; del buf7  # reuse
        buf11 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_14, x_19, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf10, arg172_1, arg173_1, arg4_1, arg5_1, buf9, arg174_1, arg175_1, arg6_1, arg7_1, buf11, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf10
        del buf9
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_28], Original ATen: [aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg126_1
        buf13 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg127_1, buf13, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg127_1
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf11, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del buf13
        buf15 = buf12; del buf12  # reuse
        # Source Nodes: [x_29, x_34, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf15, arg178_1, arg179_1, arg10_1, arg11_1, buf14, arg180_1, arg181_1, arg12_1, arg13_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg178_1
        del arg179_1
        del arg180_1
        del arg181_1
        del buf14
        buf16 = buf11; del buf11  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____1___act, x_25, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf16, buf15, arg176_1, arg177_1, arg8_1, arg9_1, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del arg176_1
        del arg177_1
        del arg8_1
        del arg9_1
        del buf15
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg128_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg128_1
        buf18 = empty_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(arg129_1, buf18, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg129_1
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf16, buf18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 192, 28, 28), (150528, 784, 28, 1))
        del buf16
        del buf18
        buf20 = buf17; del buf17  # reuse
        buf21 = reinterpret_tensor(buf2, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf2  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_43, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf20, arg182_1, arg183_1, arg14_1, arg15_1, buf19, arg184_1, arg185_1, arg16_1, arg17_1, buf21, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf19
        del buf20
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_57], Original ATen: [aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg130_1
        buf23 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(arg131_1, buf23, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg131_1
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf21, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf25 = buf22; del buf22  # reuse
        # Source Nodes: [x_58, x_63, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf25, arg188_1, arg189_1, arg20_1, arg21_1, buf24, arg190_1, arg191_1, arg22_1, arg23_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        del buf24
        buf26 = buf21; del buf21  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act, x_54, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf26, buf25, arg186_1, arg187_1, arg18_1, arg19_1, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg186_1
        del arg187_1
        del arg18_1
        del arg19_1
        del buf25
        # Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg132_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(arg133_1, buf28, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg133_1
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf26, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf30 = buf27; del buf27  # reuse
        # Source Nodes: [x_75, x_80, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf30, arg194_1, arg195_1, arg26_1, arg27_1, buf29, arg196_1, arg197_1, arg28_1, arg29_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del arg197_1
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del buf29
        buf31 = buf26; del buf26  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____2___act, x_71, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf31, buf30, arg192_1, arg193_1, arg24_1, arg25_1, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg192_1
        del arg193_1
        del arg24_1
        del arg25_1
        del buf30
        # Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg134_1
        buf33 = buf28; del buf28  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(arg135_1, buf33, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg135_1
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf31, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 784, 28, 1))
        del buf33
        buf35 = buf32; del buf32  # reuse
        # Source Nodes: [x_101, x_92, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf35, arg200_1, arg201_1, arg32_1, arg33_1, buf34, arg202_1, arg203_1, arg34_1, arg35_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf34
        buf36 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____3___act, x_103, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf36, buf35, arg198_1, arg199_1, arg30_1, arg31_1, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg198_1
        del arg199_1
        del arg30_1
        del arg31_1
        del buf35
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg136_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg136_1
        buf38 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg137_1, buf38, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg137_1
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf36, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 384, 14, 14), (75264, 196, 14, 1))
        del buf36
        del buf38
        buf40 = buf37; del buf37  # reuse
        buf41 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_106, x_111, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf40, arg204_1, arg205_1, arg36_1, arg37_1, buf39, arg206_1, arg207_1, arg38_1, arg39_1, buf41, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg204_1
        del arg205_1
        del arg206_1
        del arg207_1
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        del buf39
        del buf40
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_120], Original ATen: [aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg138_1
        buf43 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg139_1, buf43, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg139_1
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf41, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf45 = buf42; del buf42  # reuse
        # Source Nodes: [x_121, x_126, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf45, arg210_1, arg211_1, arg42_1, arg43_1, buf44, arg212_1, arg213_1, arg44_1, arg45_1, 602112, grid=grid(602112), stream=stream0)
        del arg210_1
        del arg211_1
        del arg212_1
        del arg213_1
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf44
        buf46 = buf41; del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act, x_117, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf46, buf45, arg208_1, arg209_1, arg40_1, arg41_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg208_1
        del arg209_1
        del arg40_1
        del arg41_1
        del buf45
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg140_1
        buf48 = buf43; del buf43  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg141_1, buf48, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg141_1
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf46, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf50 = buf47; del buf47  # reuse
        # Source Nodes: [x_138, x_143, x_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf50, arg216_1, arg217_1, arg48_1, arg49_1, buf49, arg218_1, arg219_1, arg50_1, arg51_1, 602112, grid=grid(602112), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        del buf49
        buf51 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act, x_134, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf51, buf50, arg214_1, arg215_1, arg46_1, arg47_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg214_1
        del arg215_1
        del arg46_1
        del arg47_1
        del buf50
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg142_1
        buf53 = buf48; del buf48  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg143_1, buf53, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg143_1
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf51, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf55 = buf52; del buf52  # reuse
        # Source Nodes: [x_155, x_160, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf55, arg222_1, arg223_1, arg54_1, arg55_1, buf54, arg224_1, arg225_1, arg56_1, arg57_1, 602112, grid=grid(602112), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        del buf54
        buf56 = buf51; del buf51  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act, x_151, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf56, buf55, arg220_1, arg221_1, arg52_1, arg53_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg220_1
        del arg221_1
        del arg52_1
        del arg53_1
        del buf55
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg144_1
        buf58 = buf53; del buf53  # reuse
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg145_1, buf58, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg145_1
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf56, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf60 = buf57; del buf57  # reuse
        # Source Nodes: [x_172, x_177, x_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf60, arg228_1, arg229_1, arg60_1, arg61_1, buf59, arg230_1, arg231_1, arg62_1, arg63_1, 602112, grid=grid(602112), stream=stream0)
        del arg228_1
        del arg229_1
        del arg230_1
        del arg231_1
        del arg60_1
        del arg61_1
        del arg62_1
        del arg63_1
        del buf59
        buf61 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act, x_168, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf61, buf60, arg226_1, arg227_1, arg58_1, arg59_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg226_1
        del arg227_1
        del arg58_1
        del arg59_1
        del buf60
        # Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg146_1
        buf63 = buf58; del buf58  # reuse
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg147_1, buf63, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg147_1
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf61, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf65 = buf62; del buf62  # reuse
        # Source Nodes: [x_189, x_194, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf65, arg234_1, arg235_1, arg66_1, arg67_1, buf64, arg236_1, arg237_1, arg68_1, arg69_1, 602112, grid=grid(602112), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del arg237_1
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        del buf64
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act, x_185, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf66, buf65, arg232_1, arg233_1, arg64_1, arg65_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg232_1
        del arg233_1
        del arg64_1
        del arg65_1
        del buf65
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg148_1
        buf68 = buf63; del buf63  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg149_1, buf68, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg149_1
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf66, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf70 = buf67; del buf67  # reuse
        # Source Nodes: [x_206, x_211, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf70, arg240_1, arg241_1, arg72_1, arg73_1, buf69, arg242_1, arg243_1, arg74_1, arg75_1, 602112, grid=grid(602112), stream=stream0)
        del arg240_1
        del arg241_1
        del arg242_1
        del arg243_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf69
        buf71 = buf66; del buf66  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____6___act, x_202, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf71, buf70, arg238_1, arg239_1, arg70_1, arg71_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg238_1
        del arg239_1
        del arg70_1
        del arg71_1
        del buf70
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg150_1
        buf73 = buf68; del buf68  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg151_1, buf73, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg151_1
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf71, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf75 = buf72; del buf72  # reuse
        # Source Nodes: [x_223, x_228, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf75, arg246_1, arg247_1, arg78_1, arg79_1, buf74, arg248_1, arg249_1, arg80_1, arg81_1, 602112, grid=grid(602112), stream=stream0)
        del arg246_1
        del arg247_1
        del arg248_1
        del arg249_1
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del buf74
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____7___act, x_219, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf76, buf75, arg244_1, arg245_1, arg76_1, arg77_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg244_1
        del arg245_1
        del arg76_1
        del arg77_1
        del buf75
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg152_1
        buf78 = buf73; del buf73  # reuse
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg153_1, buf78, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg153_1
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf76, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf80 = buf77; del buf77  # reuse
        # Source Nodes: [x_240, x_245, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf80, arg252_1, arg253_1, arg84_1, arg85_1, buf79, arg254_1, arg255_1, arg86_1, arg87_1, 602112, grid=grid(602112), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        del buf79
        buf81 = buf76; del buf76  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____8___act, x_236, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf81, buf80, arg250_1, arg251_1, arg82_1, arg83_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg250_1
        del arg251_1
        del arg82_1
        del arg83_1
        del buf80
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg154_1
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg155_1, buf83, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg155_1
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf81, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf85 = buf82; del buf82  # reuse
        # Source Nodes: [x_257, x_262, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf85, arg258_1, arg259_1, arg90_1, arg91_1, buf84, arg260_1, arg261_1, arg92_1, arg93_1, 602112, grid=grid(602112), stream=stream0)
        del arg258_1
        del arg259_1
        del arg260_1
        del arg261_1
        del arg90_1
        del arg91_1
        del arg92_1
        del arg93_1
        del buf84
        buf86 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____9___act, x_253, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf86, buf85, arg256_1, arg257_1, arg88_1, arg89_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg256_1
        del arg257_1
        del arg88_1
        del arg89_1
        del buf85
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg156_1
        buf88 = buf83; del buf83  # reuse
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg157_1, buf88, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg157_1
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf86, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf90 = buf87; del buf87  # reuse
        # Source Nodes: [x_274, x_279, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf90, arg264_1, arg265_1, arg96_1, arg97_1, buf89, arg266_1, arg267_1, arg98_1, arg99_1, 602112, grid=grid(602112), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf89
        buf91 = buf86; del buf86  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____10___act, x_270, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf91, buf90, arg262_1, arg263_1, arg94_1, arg95_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg262_1
        del arg263_1
        del arg94_1
        del arg95_1
        del buf90
        # Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg158_1
        buf93 = buf88; del buf88  # reuse
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg159_1, buf93, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg159_1
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf91, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf95 = buf92; del buf92  # reuse
        # Source Nodes: [x_291, x_296, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf95, arg270_1, arg271_1, arg102_1, arg103_1, buf94, arg272_1, arg273_1, arg104_1, arg105_1, 602112, grid=grid(602112), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del arg270_1
        del arg271_1
        del arg272_1
        del arg273_1
        del buf94
        buf96 = buf91; del buf91  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____11___act, x_287, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf96, buf95, arg268_1, arg269_1, arg100_1, arg101_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg100_1
        del arg101_1
        del arg268_1
        del arg269_1
        del buf95
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg160_1
        buf98 = buf93; del buf93  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg161_1, buf98, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg161_1
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf96, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf100 = buf97; del buf97  # reuse
        # Source Nodes: [x_308, x_313, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf100, arg276_1, arg277_1, arg108_1, arg109_1, buf99, arg278_1, arg279_1, arg110_1, arg111_1, 602112, grid=grid(602112), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del arg111_1
        del arg276_1
        del arg277_1
        del arg278_1
        del arg279_1
        del buf99
        buf101 = buf96; del buf96  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____12___act, x_304, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf101, buf100, arg274_1, arg275_1, arg106_1, arg107_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg106_1
        del arg107_1
        del arg274_1
        del arg275_1
        del buf100
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg162_1
        buf103 = buf98; del buf98  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg163_1, buf103, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg163_1
        # Source Nodes: [x_329], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf101, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 384, 14, 14), (75264, 196, 14, 1))
        del buf103
        buf105 = buf102; del buf102  # reuse
        # Source Nodes: [x_325, x_330, x_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf105, arg282_1, arg283_1, arg114_1, arg115_1, buf104, arg284_1, arg285_1, arg116_1, arg117_1, 602112, grid=grid(602112), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf104
        buf106 = buf101; del buf101  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____13___act, x_321, x_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf106, buf105, arg280_1, arg281_1, arg112_1, arg113_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg112_1
        del arg113_1
        del arg280_1
        del arg281_1
        del buf105
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg164_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 1408, 7, 7), (68992, 49, 7, 1))
        del arg164_1
        buf108 = empty_strided((1408, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg165_1, buf108, 540672, 9, grid=grid(540672, 9), stream=stream0)
        del arg165_1
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf106, buf108, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 1408, 7, 7), (68992, 49, 7, 1))
        del buf106
        del buf108
        buf110 = buf107; del buf107  # reuse
        buf111 = empty_strided((8, 1408, 1, 1), (1408, 1, 11264, 11264), device='cuda', dtype=torch.float32)
        buf112 = reinterpret_tensor(buf111, (8, 1408, 1, 1), (1408, 1, 1, 1), 0); del buf111  # reuse
        # Source Nodes: [x_339, x_344, x_348, x_350, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_19.run(buf110, buf112, arg286_1, arg287_1, arg118_1, arg119_1, buf109, arg288_1, arg289_1, arg120_1, arg121_1, 11264, 49, grid=grid(11264), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        del arg121_1
        del arg286_1
        del arg287_1
        del arg288_1
        del arg289_1
        del buf109
        del buf110
        buf113 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg167_1, reinterpret_tensor(buf112, (8, 1408), (1408, 1), 0), reinterpret_tensor(arg166_1, (1408, 1000), (1, 1408), 0), alpha=1, beta=1, out=buf113)
        del arg166_1
        del arg167_1
        return (buf113, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1408, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
