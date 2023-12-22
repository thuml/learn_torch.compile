
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrwkxcpboofanir4ahdijl45q63dz5oiulxumeqxm5cu3ocy5kg.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbxnrzrqop4klqko3lxrugnst7kip4kb4m4tqgyf3ww2kcgb2l.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4ocxojhpfqxs2jqvopqv4urlwexlh7kew5awi62hv27uqhxuzk.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (32*x2) + (524288*y1)), tmp15, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftrkl5p66kuxddkad67iw2ucegbocng6quu6httmbmopjgblfbr.py
# Source Nodes: [x_6], Original ATen: [aten.convolution]
# x_6 => convolution_1
triton_poi_fused_convolution_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': []},
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
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7v7psyjk5j4bhacsqmh2sp3kfpygkeyegmnkxbgf6rp7hhrajg4.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_11 => relu_1
# x_7 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqo4zdi7bdwheywe2zfmlekvr3t5ympyek7drp2k6ifq3humrfvb.py
# Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_11 => relu_1
# x_12 => convolution_2
# x_7 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/56/c56ceevrgruxlximpkjlkfcxismtojfkte6fx332sidxwqsilzcn.py
# Source Nodes: [shortcut_1, x_13, x_21, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_1 => relu_2
# x_13 => add_5, mul_7, mul_8, sub_2
# x_21 => add_7, mul_10, mul_11, sub_3
# x_25 => add_8
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbe76iownbt5yikuocluts4zudmhkjx2bbk43wdafzmyj343obcc.py
# Source Nodes: [shortcut_1, x_26], Original ATen: [aten.convolution, aten.relu]
# shortcut_1 => relu_2
# x_26 => convolution_4
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
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


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7czkrphmbqkbjgkp2ukojjpu4vcn6ttgs3qxjbthqe7yzlcguu.py
# Source Nodes: [x_27, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_27 => add_10, mul_13, mul_14, sub_4
# x_31 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x2) + (196608*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofeneixv46m6xk426zj34zpervzwaxnktxs7cz7jdq53mfotqvg.py
# Source Nodes: [x_27, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_27 => add_10, mul_13, mul_14, sub_4
# x_31 => relu_3
# x_32 => convolution_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwldwgadhwcfzo5yv3x3qo72r5dxkglswgdcze2qnp233e2jnee.py
# Source Nodes: [shortcut_2, x_33, x_41, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_2 => relu_4
# x_33 => add_12, mul_16, mul_17, sub_5
# x_41 => add_14, mul_19, mul_20, sub_6
# x_45 => add_15
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (196608*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnxwgy4yqgbecfr2yuupmbngpzedsmanvncmpju7l2p2x64buir.py
# Source Nodes: [shortcut_3, x_53, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_6
# x_53 => add_19, mul_25, mul_26, sub_8
# x_60 => add_20
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capcosokwnc2xmtpzomipau3yczu3r6ig6vwjuihz6xtd5k43hrf.py
# Source Nodes: [x_62, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_62 => add_22, mul_28, mul_29, sub_9
# x_66 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (160*x2) + (163840*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygdkaahk3yixgwlidki24f6uxgwiahcgusk46myuxosbes5n3ad.py
# Source Nodes: [x_62, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_62 => add_22, mul_28, mul_29, sub_9
# x_66 => relu_7
# x_67 => convolution_10
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1440*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkgc2phmtqmajtybli5x4kxgmv4qkloxnqibb3hharoir6ovphz.py
# Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_68 => add_24, mul_31, mul_32, sub_10
# x_72 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (160*x2) + (40960*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crptklumgfh4iigx54vyymof5bwus5wl5vauzpmisfsbnepdb7vp.py
# Source Nodes: [shortcut_4, x_76, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_4 => relu_9
# x_76 => add_26, mul_34, mul_35, sub_11
# x_84 => add_28, mul_37, mul_38, sub_12
# x_88 => add_29
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (640*x2) + (163840*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftzhglbe5f3bqiz32tzu5edhzod6wt6wdtt54g6ye2wly4p4zfx.py
# Source Nodes: [shortcut_5, x_104, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_5 => relu_12
# x_104 => add_35, mul_46, mul_47, sub_15
# x_111 => add_36
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (640*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp54ntym2bp6acpug6hdiqh7nk56pig4ojmjtxx6k3n7adsqjclz.py
# Source Nodes: [x_205, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_205 => add_66, mul_85, mul_86, sub_28
# x_209 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1920
    y1 = (yindex // 1920)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (1920*x2) + (491520*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cek4bdn2bjhlbvg2i54y3a7kzndxdvudsyd5ieczfmydwavccqji.py
# Source Nodes: [x_211, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_211 => add_68, mul_88, mul_89, sub_29
# x_215 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1920
    y1 = (yindex // 1920)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (1920*x2) + (122880*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnphssajdxkwiugqnx3thti5p7cblkkpsunsyr4uet3bnvkmlgis.py
# Source Nodes: [shortcut_10, x_219, x_227, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_10 => relu_27
# x_219 => add_70, mul_91, mul_92, sub_30
# x_227 => add_72, mul_94, mul_95, sub_31
# x_231 => add_73
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (640*x2) + (40960*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cneabkbbdxmbz3ev6dibkhqkehxejboy4gm4imrt4gglnbyaswqb.py
# Source Nodes: [shortcut_11, x_247, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_11 => relu_30
# x_247 => add_79, mul_103, mul_104, sub_34
# x_254 => add_80
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (640*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2ezyhfopqfjtyrdmkcmbr75l6tonm5yafuof3cuvruuzfhv6jb.py
# Source Nodes: [x_418, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_418 => add_131, mul_169, mul_170, sub_56
# x_423 => relu_52
# x_424 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2560
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 64.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (192, ), (1, ))
    assert_size_stride(arg11_1, (192, ), (1, ))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (192, ), (1, ))
    assert_size_stride(arg18_1, (160, ), (1, ))
    assert_size_stride(arg19_1, (160, ), (1, ))
    assert_size_stride(arg20_1, (160, ), (1, ))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (640, ), (1, ))
    assert_size_stride(arg23_1, (640, ), (1, ))
    assert_size_stride(arg24_1, (640, ), (1, ))
    assert_size_stride(arg25_1, (640, ), (1, ))
    assert_size_stride(arg26_1, (160, ), (1, ))
    assert_size_stride(arg27_1, (160, ), (1, ))
    assert_size_stride(arg28_1, (160, ), (1, ))
    assert_size_stride(arg29_1, (160, ), (1, ))
    assert_size_stride(arg30_1, (640, ), (1, ))
    assert_size_stride(arg31_1, (640, ), (1, ))
    assert_size_stride(arg32_1, (160, ), (1, ))
    assert_size_stride(arg33_1, (160, ), (1, ))
    assert_size_stride(arg34_1, (160, ), (1, ))
    assert_size_stride(arg35_1, (160, ), (1, ))
    assert_size_stride(arg36_1, (640, ), (1, ))
    assert_size_stride(arg37_1, (640, ), (1, ))
    assert_size_stride(arg38_1, (160, ), (1, ))
    assert_size_stride(arg39_1, (160, ), (1, ))
    assert_size_stride(arg40_1, (160, ), (1, ))
    assert_size_stride(arg41_1, (160, ), (1, ))
    assert_size_stride(arg42_1, (640, ), (1, ))
    assert_size_stride(arg43_1, (640, ), (1, ))
    assert_size_stride(arg44_1, (160, ), (1, ))
    assert_size_stride(arg45_1, (160, ), (1, ))
    assert_size_stride(arg46_1, (160, ), (1, ))
    assert_size_stride(arg47_1, (160, ), (1, ))
    assert_size_stride(arg48_1, (640, ), (1, ))
    assert_size_stride(arg49_1, (640, ), (1, ))
    assert_size_stride(arg50_1, (160, ), (1, ))
    assert_size_stride(arg51_1, (160, ), (1, ))
    assert_size_stride(arg52_1, (160, ), (1, ))
    assert_size_stride(arg53_1, (160, ), (1, ))
    assert_size_stride(arg54_1, (640, ), (1, ))
    assert_size_stride(arg55_1, (640, ), (1, ))
    assert_size_stride(arg56_1, (1920, ), (1, ))
    assert_size_stride(arg57_1, (1920, ), (1, ))
    assert_size_stride(arg58_1, (1920, ), (1, ))
    assert_size_stride(arg59_1, (1920, ), (1, ))
    assert_size_stride(arg60_1, (640, ), (1, ))
    assert_size_stride(arg61_1, (640, ), (1, ))
    assert_size_stride(arg62_1, (640, ), (1, ))
    assert_size_stride(arg63_1, (640, ), (1, ))
    assert_size_stride(arg64_1, (1920, ), (1, ))
    assert_size_stride(arg65_1, (1920, ), (1, ))
    assert_size_stride(arg66_1, (1920, ), (1, ))
    assert_size_stride(arg67_1, (1920, ), (1, ))
    assert_size_stride(arg68_1, (640, ), (1, ))
    assert_size_stride(arg69_1, (640, ), (1, ))
    assert_size_stride(arg70_1, (1920, ), (1, ))
    assert_size_stride(arg71_1, (1920, ), (1, ))
    assert_size_stride(arg72_1, (1920, ), (1, ))
    assert_size_stride(arg73_1, (1920, ), (1, ))
    assert_size_stride(arg74_1, (640, ), (1, ))
    assert_size_stride(arg75_1, (640, ), (1, ))
    assert_size_stride(arg76_1, (1920, ), (1, ))
    assert_size_stride(arg77_1, (1920, ), (1, ))
    assert_size_stride(arg78_1, (1920, ), (1, ))
    assert_size_stride(arg79_1, (1920, ), (1, ))
    assert_size_stride(arg80_1, (640, ), (1, ))
    assert_size_stride(arg81_1, (640, ), (1, ))
    assert_size_stride(arg82_1, (1920, ), (1, ))
    assert_size_stride(arg83_1, (1920, ), (1, ))
    assert_size_stride(arg84_1, (1920, ), (1, ))
    assert_size_stride(arg85_1, (1920, ), (1, ))
    assert_size_stride(arg86_1, (640, ), (1, ))
    assert_size_stride(arg87_1, (640, ), (1, ))
    assert_size_stride(arg88_1, (1920, ), (1, ))
    assert_size_stride(arg89_1, (1920, ), (1, ))
    assert_size_stride(arg90_1, (1920, ), (1, ))
    assert_size_stride(arg91_1, (1920, ), (1, ))
    assert_size_stride(arg92_1, (640, ), (1, ))
    assert_size_stride(arg93_1, (640, ), (1, ))
    assert_size_stride(arg94_1, (1920, ), (1, ))
    assert_size_stride(arg95_1, (1920, ), (1, ))
    assert_size_stride(arg96_1, (1920, ), (1, ))
    assert_size_stride(arg97_1, (1920, ), (1, ))
    assert_size_stride(arg98_1, (640, ), (1, ))
    assert_size_stride(arg99_1, (640, ), (1, ))
    assert_size_stride(arg100_1, (1920, ), (1, ))
    assert_size_stride(arg101_1, (1920, ), (1, ))
    assert_size_stride(arg102_1, (1920, ), (1, ))
    assert_size_stride(arg103_1, (1920, ), (1, ))
    assert_size_stride(arg104_1, (640, ), (1, ))
    assert_size_stride(arg105_1, (640, ), (1, ))
    assert_size_stride(arg106_1, (1920, ), (1, ))
    assert_size_stride(arg107_1, (1920, ), (1, ))
    assert_size_stride(arg108_1, (1920, ), (1, ))
    assert_size_stride(arg109_1, (1920, ), (1, ))
    assert_size_stride(arg110_1, (640, ), (1, ))
    assert_size_stride(arg111_1, (640, ), (1, ))
    assert_size_stride(arg112_1, (2560, ), (1, ))
    assert_size_stride(arg113_1, (2560, ), (1, ))
    assert_size_stride(arg114_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg115_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg116_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg117_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg118_1, (192, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg119_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg120_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg121_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg122_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg123_1, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg124_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg125_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg126_1, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg127_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg128_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg129_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg130_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg131_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg132_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg133_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg134_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg135_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg136_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg137_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg138_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg139_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg140_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg141_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg142_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg143_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg144_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg145_1, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg146_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg147_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg149_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg150_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg152_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg153_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg155_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg156_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg157_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg158_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg159_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg161_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg162_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg163_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg164_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg165_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg166_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg167_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg168_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg169_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg170_1, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg171_1, (1000, 2560), (2560, 1))
    assert_size_stride(arg172_1, (1000, ), (1, ))
    assert_size_stride(arg173_1, (32, ), (1, ))
    assert_size_stride(arg174_1, (32, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (192, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, ), (1, ))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, ), (1, ))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (160, ), (1, ))
    assert_size_stride(arg192_1, (160, ), (1, ))
    assert_size_stride(arg193_1, (160, ), (1, ))
    assert_size_stride(arg194_1, (160, ), (1, ))
    assert_size_stride(arg195_1, (640, ), (1, ))
    assert_size_stride(arg196_1, (640, ), (1, ))
    assert_size_stride(arg197_1, (640, ), (1, ))
    assert_size_stride(arg198_1, (640, ), (1, ))
    assert_size_stride(arg199_1, (160, ), (1, ))
    assert_size_stride(arg200_1, (160, ), (1, ))
    assert_size_stride(arg201_1, (160, ), (1, ))
    assert_size_stride(arg202_1, (160, ), (1, ))
    assert_size_stride(arg203_1, (640, ), (1, ))
    assert_size_stride(arg204_1, (640, ), (1, ))
    assert_size_stride(arg205_1, (160, ), (1, ))
    assert_size_stride(arg206_1, (160, ), (1, ))
    assert_size_stride(arg207_1, (160, ), (1, ))
    assert_size_stride(arg208_1, (160, ), (1, ))
    assert_size_stride(arg209_1, (640, ), (1, ))
    assert_size_stride(arg210_1, (640, ), (1, ))
    assert_size_stride(arg211_1, (160, ), (1, ))
    assert_size_stride(arg212_1, (160, ), (1, ))
    assert_size_stride(arg213_1, (160, ), (1, ))
    assert_size_stride(arg214_1, (160, ), (1, ))
    assert_size_stride(arg215_1, (640, ), (1, ))
    assert_size_stride(arg216_1, (640, ), (1, ))
    assert_size_stride(arg217_1, (160, ), (1, ))
    assert_size_stride(arg218_1, (160, ), (1, ))
    assert_size_stride(arg219_1, (160, ), (1, ))
    assert_size_stride(arg220_1, (160, ), (1, ))
    assert_size_stride(arg221_1, (640, ), (1, ))
    assert_size_stride(arg222_1, (640, ), (1, ))
    assert_size_stride(arg223_1, (160, ), (1, ))
    assert_size_stride(arg224_1, (160, ), (1, ))
    assert_size_stride(arg225_1, (160, ), (1, ))
    assert_size_stride(arg226_1, (160, ), (1, ))
    assert_size_stride(arg227_1, (640, ), (1, ))
    assert_size_stride(arg228_1, (640, ), (1, ))
    assert_size_stride(arg229_1, (1920, ), (1, ))
    assert_size_stride(arg230_1, (1920, ), (1, ))
    assert_size_stride(arg231_1, (1920, ), (1, ))
    assert_size_stride(arg232_1, (1920, ), (1, ))
    assert_size_stride(arg233_1, (640, ), (1, ))
    assert_size_stride(arg234_1, (640, ), (1, ))
    assert_size_stride(arg235_1, (640, ), (1, ))
    assert_size_stride(arg236_1, (640, ), (1, ))
    assert_size_stride(arg237_1, (1920, ), (1, ))
    assert_size_stride(arg238_1, (1920, ), (1, ))
    assert_size_stride(arg239_1, (1920, ), (1, ))
    assert_size_stride(arg240_1, (1920, ), (1, ))
    assert_size_stride(arg241_1, (640, ), (1, ))
    assert_size_stride(arg242_1, (640, ), (1, ))
    assert_size_stride(arg243_1, (1920, ), (1, ))
    assert_size_stride(arg244_1, (1920, ), (1, ))
    assert_size_stride(arg245_1, (1920, ), (1, ))
    assert_size_stride(arg246_1, (1920, ), (1, ))
    assert_size_stride(arg247_1, (640, ), (1, ))
    assert_size_stride(arg248_1, (640, ), (1, ))
    assert_size_stride(arg249_1, (1920, ), (1, ))
    assert_size_stride(arg250_1, (1920, ), (1, ))
    assert_size_stride(arg251_1, (1920, ), (1, ))
    assert_size_stride(arg252_1, (1920, ), (1, ))
    assert_size_stride(arg253_1, (640, ), (1, ))
    assert_size_stride(arg254_1, (640, ), (1, ))
    assert_size_stride(arg255_1, (1920, ), (1, ))
    assert_size_stride(arg256_1, (1920, ), (1, ))
    assert_size_stride(arg257_1, (1920, ), (1, ))
    assert_size_stride(arg258_1, (1920, ), (1, ))
    assert_size_stride(arg259_1, (640, ), (1, ))
    assert_size_stride(arg260_1, (640, ), (1, ))
    assert_size_stride(arg261_1, (1920, ), (1, ))
    assert_size_stride(arg262_1, (1920, ), (1, ))
    assert_size_stride(arg263_1, (1920, ), (1, ))
    assert_size_stride(arg264_1, (1920, ), (1, ))
    assert_size_stride(arg265_1, (640, ), (1, ))
    assert_size_stride(arg266_1, (640, ), (1, ))
    assert_size_stride(arg267_1, (1920, ), (1, ))
    assert_size_stride(arg268_1, (1920, ), (1, ))
    assert_size_stride(arg269_1, (1920, ), (1, ))
    assert_size_stride(arg270_1, (1920, ), (1, ))
    assert_size_stride(arg271_1, (640, ), (1, ))
    assert_size_stride(arg272_1, (640, ), (1, ))
    assert_size_stride(arg273_1, (1920, ), (1, ))
    assert_size_stride(arg274_1, (1920, ), (1, ))
    assert_size_stride(arg275_1, (1920, ), (1, ))
    assert_size_stride(arg276_1, (1920, ), (1, ))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (640, ), (1, ))
    assert_size_stride(arg279_1, (1920, ), (1, ))
    assert_size_stride(arg280_1, (1920, ), (1, ))
    assert_size_stride(arg281_1, (1920, ), (1, ))
    assert_size_stride(arg282_1, (1920, ), (1, ))
    assert_size_stride(arg283_1, (640, ), (1, ))
    assert_size_stride(arg284_1, (640, ), (1, ))
    assert_size_stride(arg285_1, (2560, ), (1, ))
    assert_size_stride(arg286_1, (2560, ), (1, ))
    assert_size_stride(arg287_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg287_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg287_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg114_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg114_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del buf1
        buf3 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg173_1, arg174_1, arg0_1, arg1_1, buf3, 256, 16384, grid=grid(256, 16384), stream=stream0)
        del arg0_1
        del arg173_1
        del arg174_1
        del arg1_1
        buf4 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg115_1, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg115_1
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf4
        buf6 = reinterpret_tensor(buf2, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf2  # reuse
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, arg175_1, arg176_1, arg2_1, arg3_1, buf6, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del arg175_1
        del arg176_1
        del arg2_1
        del arg3_1
        del buf5
        buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg116_1, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg116_1
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf6
        del buf7
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf3, arg117_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg117_1
        buf10 = buf8; del buf8  # reuse
        buf11 = reinterpret_tensor(buf3, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf3  # reuse
        # Source Nodes: [shortcut_1, x_13, x_21, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf10, arg177_1, arg178_1, arg4_1, arg5_1, buf9, arg179_1, arg180_1, arg6_1, arg7_1, buf11, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf10
        del buf9
        buf12 = empty_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1, x_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_7.run(arg118_1, buf12, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg118_1
        # Source Nodes: [shortcut_1, x_26], Original ATen: [aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 192, 32, 32), (196608, 1024, 32, 1))
        del buf12
        buf14 = reinterpret_tensor(buf0, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf0  # reuse
        # Source Nodes: [x_27, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf13, arg181_1, arg182_1, arg8_1, arg9_1, buf14, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        del arg181_1
        del arg182_1
        del arg8_1
        del arg9_1
        del buf13
        buf15 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg119_1, buf15, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg119_1
        # Source Nodes: [x_27, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 192, 32, 32), (196608, 1024, 32, 1))
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf11, arg120_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 192, 32, 32), (196608, 1024, 32, 1))
        del arg120_1
        del buf11
        buf18 = buf16; del buf16  # reuse
        buf19 = buf14; del buf14  # reuse
        # Source Nodes: [shortcut_2, x_33, x_41, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf18, arg183_1, arg184_1, arg10_1, arg11_1, buf17, arg185_1, arg186_1, arg12_1, arg13_1, buf19, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del buf17
        buf20 = buf15; del buf15  # reuse
        # Source Nodes: [shortcut_2, x_46], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg121_1, buf20, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg121_1
        # Source Nodes: [shortcut_2, x_46], Original ATen: [aten.convolution, aten.relu]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf22 = reinterpret_tensor(buf18, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf18  # reuse
        # Source Nodes: [x_47, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf21, arg187_1, arg188_1, arg14_1, arg15_1, buf22, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        del arg14_1
        del arg15_1
        del arg187_1
        del arg188_1
        del buf21
        buf23 = buf20; del buf20  # reuse
        # Source Nodes: [x_47, x_51, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg122_1, buf23, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg122_1
        # Source Nodes: [x_47, x_51, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 192, 32, 32), (196608, 1024, 32, 1))
        del buf22
        del buf23
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [shortcut_3, x_53, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf25, buf24, arg189_1, arg190_1, arg16_1, arg17_1, 8192, 192, grid=grid(8192, 192), stream=stream0)
        del arg16_1
        del arg17_1
        del arg189_1
        del arg190_1
        del buf24
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 160, 32, 32), (163840, 1024, 32, 1))
        del arg123_1
        buf27 = empty_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf26, arg191_1, arg192_1, arg18_1, arg19_1, buf27, 1280, 1024, grid=grid(1280, 1024), stream=stream0)
        del arg18_1
        del arg191_1
        del arg192_1
        del arg19_1
        del buf26
        buf28 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg124_1, buf28, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg124_1
        # Source Nodes: [x_62, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf30 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf29, arg193_1, arg194_1, arg20_1, arg21_1, buf30, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg193_1
        del arg194_1
        del arg20_1
        del arg21_1
        del buf29
        # Source Nodes: [x_68, x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf31 = extern_kernels.convolution(buf30, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg125_1
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf25, arg126_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg126_1
        del buf25
        buf33 = buf31; del buf31  # reuse
        buf34 = reinterpret_tensor(buf27, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf27  # reuse
        # Source Nodes: [shortcut_4, x_76, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf33, arg195_1, arg196_1, arg22_1, arg23_1, buf32, arg197_1, arg198_1, arg24_1, arg25_1, buf34, 5120, 256, grid=grid(5120, 256), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf32
        del buf33
        # Source Nodes: [shortcut_4, x_89], Original ATen: [aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 160, 16, 16), (40960, 256, 16, 1))
        del arg127_1
        buf36 = buf30; del buf30  # reuse
        # Source Nodes: [x_90, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf35, arg199_1, arg200_1, arg26_1, arg27_1, buf36, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg199_1
        del arg200_1
        del arg26_1
        del arg27_1
        del buf35
        buf37 = buf28; del buf28  # reuse
        # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg128_1, buf37, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg128_1
        # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf39 = buf36; del buf36  # reuse
        # Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf38, arg201_1, arg202_1, arg28_1, arg29_1, buf39, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg201_1
        del arg202_1
        del arg28_1
        del arg29_1
        del buf38
        # Source Nodes: [x_100, x_103, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf39, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg129_1
        buf41 = buf34; del buf34  # reuse
        # Source Nodes: [shortcut_5, x_104, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf41, buf40, arg203_1, arg204_1, arg30_1, arg31_1, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del arg203_1
        del arg204_1
        del arg30_1
        del arg31_1
        del buf40
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 160, 16, 16), (40960, 256, 16, 1))
        del arg130_1
        buf43 = buf39; del buf39  # reuse
        # Source Nodes: [x_113, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf42, arg205_1, arg206_1, arg32_1, arg33_1, buf43, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg205_1
        del arg206_1
        del arg32_1
        del arg33_1
        del buf42
        buf44 = buf37; del buf37  # reuse
        # Source Nodes: [x_113, x_117, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg131_1, buf44, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg131_1
        # Source Nodes: [x_113, x_117, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf46 = buf43; del buf43  # reuse
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf45, arg207_1, arg208_1, arg34_1, arg35_1, buf46, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg207_1
        del arg208_1
        del arg34_1
        del arg35_1
        del buf45
        # Source Nodes: [x_119, x_123, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf46, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg132_1
        buf48 = buf41; del buf41  # reuse
        # Source Nodes: [shortcut_6, x_127, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf48, buf47, arg209_1, arg210_1, arg36_1, arg37_1, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del arg209_1
        del arg210_1
        del arg36_1
        del arg37_1
        del buf47
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 160, 16, 16), (40960, 256, 16, 1))
        del arg133_1
        buf50 = buf46; del buf46  # reuse
        # Source Nodes: [x_136, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf49, arg211_1, arg212_1, arg38_1, arg39_1, buf50, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg211_1
        del arg212_1
        del arg38_1
        del arg39_1
        del buf49
        buf51 = buf44; del buf44  # reuse
        # Source Nodes: [x_136, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg134_1, buf51, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg134_1
        # Source Nodes: [x_136, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf53 = buf50; del buf50  # reuse
        # Source Nodes: [x_142, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf52, arg213_1, arg214_1, arg40_1, arg41_1, buf53, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg213_1
        del arg214_1
        del arg40_1
        del arg41_1
        del buf52
        # Source Nodes: [x_142, x_146, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf53, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg135_1
        buf55 = buf48; del buf48  # reuse
        # Source Nodes: [shortcut_7, x_150, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf55, buf54, arg215_1, arg216_1, arg42_1, arg43_1, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del arg215_1
        del arg216_1
        del arg42_1
        del arg43_1
        del buf54
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 160, 16, 16), (40960, 256, 16, 1))
        del arg136_1
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [x_159, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf56, arg217_1, arg218_1, arg44_1, arg45_1, buf57, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg217_1
        del arg218_1
        del arg44_1
        del arg45_1
        del buf56
        buf58 = buf51; del buf51  # reuse
        # Source Nodes: [x_159, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg137_1, buf58, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg137_1
        # Source Nodes: [x_159, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf60 = buf57; del buf57  # reuse
        # Source Nodes: [x_165, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf59, arg219_1, arg220_1, arg46_1, arg47_1, buf60, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg219_1
        del arg220_1
        del arg46_1
        del arg47_1
        del buf59
        # Source Nodes: [x_165, x_169, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf61 = extern_kernels.convolution(buf60, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg138_1
        buf62 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_8, x_173, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf62, buf61, arg221_1, arg222_1, arg48_1, arg49_1, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del arg221_1
        del arg222_1
        del arg48_1
        del arg49_1
        del buf61
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 160, 16, 16), (40960, 256, 16, 1))
        del arg139_1
        buf64 = buf60; del buf60  # reuse
        # Source Nodes: [x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf63, arg223_1, arg224_1, arg50_1, arg51_1, buf64, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg223_1
        del arg224_1
        del arg50_1
        del arg51_1
        del buf63
        buf65 = buf58; del buf58  # reuse
        # Source Nodes: [x_182, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg140_1, buf65, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg140_1
        # Source Nodes: [x_182, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 160, 16, 16), (40960, 256, 16, 1))
        del buf65
        buf67 = buf64; del buf64  # reuse
        # Source Nodes: [x_188, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf66, arg225_1, arg226_1, arg52_1, arg53_1, buf67, 1280, 256, grid=grid(1280, 256), stream=stream0)
        del arg225_1
        del arg226_1
        del arg52_1
        del arg53_1
        del buf66
        # Source Nodes: [x_188, x_192, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf67, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg141_1
        buf69 = buf62; del buf62  # reuse
        # Source Nodes: [shortcut_9, x_196, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf69, buf68, arg227_1, arg228_1, arg54_1, arg55_1, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del arg227_1
        del arg228_1
        del arg54_1
        del arg55_1
        del buf68
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 1920, 16, 16), (491520, 256, 16, 1))
        del arg142_1
        buf71 = empty_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf70, arg229_1, arg230_1, arg56_1, arg57_1, buf71, 15360, 256, grid=grid(15360, 256), stream=stream0)
        del arg229_1
        del arg230_1
        del arg56_1
        del arg57_1
        del buf70
        # Source Nodes: [x_205, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg143_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf72, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg143_1
        del buf71
        buf73 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf72, arg231_1, arg232_1, arg58_1, arg59_1, buf73, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg231_1
        del arg232_1
        del arg58_1
        del arg59_1
        del buf72
        # Source Nodes: [x_211, x_215, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg144_1
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf69, arg145_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg145_1
        del buf69
        buf76 = buf74; del buf74  # reuse
        buf77 = reinterpret_tensor(buf67, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf67  # reuse
        # Source Nodes: [shortcut_10, x_219, x_227, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf76, arg233_1, arg234_1, arg60_1, arg61_1, buf75, arg235_1, arg236_1, arg62_1, arg63_1, buf77, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del arg233_1
        del arg234_1
        del arg235_1
        del arg236_1
        del arg60_1
        del arg61_1
        del arg62_1
        del arg63_1
        del buf75
        del buf76
        # Source Nodes: [shortcut_10, x_232], Original ATen: [aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg146_1
        buf79 = buf73; del buf73  # reuse
        # Source Nodes: [x_233, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf78, arg237_1, arg238_1, arg64_1, arg65_1, buf79, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg237_1
        del arg238_1
        del arg64_1
        del arg65_1
        del buf78
        # Source Nodes: [x_233, x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf80, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg147_1
        buf81 = buf79; del buf79  # reuse
        # Source Nodes: [x_239, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf80, arg239_1, arg240_1, arg66_1, arg67_1, buf81, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg239_1
        del arg240_1
        del arg66_1
        del arg67_1
        del buf80
        # Source Nodes: [x_239, x_243, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg148_1
        buf83 = buf77; del buf77  # reuse
        # Source Nodes: [shortcut_11, x_247, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf83, buf82, arg241_1, arg242_1, arg68_1, arg69_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg241_1
        del arg242_1
        del arg68_1
        del arg69_1
        del buf82
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg149_1
        buf85 = buf81; del buf81  # reuse
        # Source Nodes: [x_256, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf84, arg243_1, arg244_1, arg70_1, arg71_1, buf85, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg243_1
        del arg244_1
        del arg70_1
        del arg71_1
        del buf84
        # Source Nodes: [x_256, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf86, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg150_1
        buf87 = buf85; del buf85  # reuse
        # Source Nodes: [x_262, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf86, arg245_1, arg246_1, arg72_1, arg73_1, buf87, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg245_1
        del arg246_1
        del arg72_1
        del arg73_1
        del buf86
        # Source Nodes: [x_262, x_266, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg151_1
        buf89 = buf83; del buf83  # reuse
        # Source Nodes: [shortcut_12, x_270, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf89, buf88, arg247_1, arg248_1, arg74_1, arg75_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg247_1
        del arg248_1
        del arg74_1
        del arg75_1
        del buf88
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg152_1
        buf91 = buf87; del buf87  # reuse
        # Source Nodes: [x_279, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf90, arg249_1, arg250_1, arg76_1, arg77_1, buf91, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg249_1
        del arg250_1
        del arg76_1
        del arg77_1
        del buf90
        # Source Nodes: [x_279, x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf91, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf92, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg153_1
        buf93 = buf91; del buf91  # reuse
        # Source Nodes: [x_285, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf92, arg251_1, arg252_1, arg78_1, arg79_1, buf93, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg251_1
        del arg252_1
        del arg78_1
        del arg79_1
        del buf92
        # Source Nodes: [x_285, x_289, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf94 = extern_kernels.convolution(buf93, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg154_1
        buf95 = buf89; del buf89  # reuse
        # Source Nodes: [shortcut_13, x_293, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf95, buf94, arg253_1, arg254_1, arg80_1, arg81_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg253_1
        del arg254_1
        del arg80_1
        del arg81_1
        del buf94
        # Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg155_1
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [x_302, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf96, arg255_1, arg256_1, arg82_1, arg83_1, buf97, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg255_1
        del arg256_1
        del arg82_1
        del arg83_1
        del buf96
        # Source Nodes: [x_302, x_306, x_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf98, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg156_1
        buf99 = buf97; del buf97  # reuse
        # Source Nodes: [x_308, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf98, arg257_1, arg258_1, arg84_1, arg85_1, buf99, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg257_1
        del arg258_1
        del arg84_1
        del arg85_1
        del buf98
        # Source Nodes: [x_308, x_312, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg157_1
        buf101 = buf95; del buf95  # reuse
        # Source Nodes: [shortcut_14, x_316, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf101, buf100, arg259_1, arg260_1, arg86_1, arg87_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg259_1
        del arg260_1
        del arg86_1
        del arg87_1
        del buf100
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg158_1
        buf103 = buf99; del buf99  # reuse
        # Source Nodes: [x_325, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf102, arg261_1, arg262_1, arg88_1, arg89_1, buf103, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg261_1
        del arg262_1
        del arg88_1
        del arg89_1
        del buf102
        # Source Nodes: [x_325, x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf104 = extern_kernels.convolution(buf103, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf104, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg159_1
        buf105 = buf103; del buf103  # reuse
        # Source Nodes: [x_331, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf104, arg263_1, arg264_1, arg90_1, arg91_1, buf105, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg263_1
        del arg264_1
        del arg90_1
        del arg91_1
        del buf104
        # Source Nodes: [x_331, x_335, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf106 = extern_kernels.convolution(buf105, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg160_1
        buf107 = buf101; del buf101  # reuse
        # Source Nodes: [shortcut_15, x_339, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf107, buf106, arg265_1, arg266_1, arg92_1, arg93_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg265_1
        del arg266_1
        del arg92_1
        del arg93_1
        del buf106
        # Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg161_1
        buf109 = buf105; del buf105  # reuse
        # Source Nodes: [x_348, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf108, arg267_1, arg268_1, arg94_1, arg95_1, buf109, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg267_1
        del arg268_1
        del arg94_1
        del arg95_1
        del buf108
        # Source Nodes: [x_348, x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf110 = extern_kernels.convolution(buf109, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf110, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg162_1
        buf111 = buf109; del buf109  # reuse
        # Source Nodes: [x_354, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf110, arg269_1, arg270_1, arg96_1, arg97_1, buf111, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg269_1
        del arg270_1
        del arg96_1
        del arg97_1
        del buf110
        # Source Nodes: [x_354, x_358, x_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg163_1
        buf113 = buf107; del buf107  # reuse
        # Source Nodes: [shortcut_16, x_362, x_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf113, buf112, arg271_1, arg272_1, arg98_1, arg99_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg271_1
        del arg272_1
        del arg98_1
        del arg99_1
        del buf112
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg164_1
        buf115 = buf111; del buf111  # reuse
        # Source Nodes: [x_371, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf114, arg273_1, arg274_1, arg100_1, arg101_1, buf115, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg100_1
        del arg101_1
        del arg273_1
        del arg274_1
        del buf114
        # Source Nodes: [x_371, x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf116 = extern_kernels.convolution(buf115, arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf116, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg165_1
        buf117 = buf115; del buf115  # reuse
        # Source Nodes: [x_377, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf116, arg275_1, arg276_1, arg102_1, arg103_1, buf117, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg102_1
        del arg103_1
        del arg275_1
        del arg276_1
        del buf116
        # Source Nodes: [x_377, x_381, x_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf118 = extern_kernels.convolution(buf117, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg166_1
        buf119 = buf113; del buf113  # reuse
        # Source Nodes: [shortcut_17, x_385, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf119, buf118, arg277_1, arg278_1, arg104_1, arg105_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg104_1
        del arg105_1
        del arg277_1
        del arg278_1
        del buf118
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg167_1
        buf121 = buf117; del buf117  # reuse
        # Source Nodes: [x_394, x_398], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf120, arg279_1, arg280_1, arg106_1, arg107_1, buf121, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg106_1
        del arg107_1
        del arg279_1
        del arg280_1
        del buf120
        # Source Nodes: [x_394, x_398, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf122 = extern_kernels.convolution(buf121, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf122, (8, 1920, 8, 8), (122880, 64, 8, 1))
        del arg168_1
        buf123 = buf121; del buf121  # reuse
        # Source Nodes: [x_400, x_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf122, arg281_1, arg282_1, arg108_1, arg109_1, buf123, 15360, 64, grid=grid(15360, 64), stream=stream0)
        del arg108_1
        del arg109_1
        del arg281_1
        del arg282_1
        del buf122
        # Source Nodes: [x_400, x_404, x_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf124 = extern_kernels.convolution(buf123, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg169_1
        del buf123
        buf125 = buf119; del buf119  # reuse
        # Source Nodes: [x_408, x_415, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf125, buf124, arg283_1, arg284_1, arg110_1, arg111_1, 512, 640, grid=grid(512, 640), stream=stream0)
        del arg110_1
        del arg111_1
        del arg283_1
        del arg284_1
        del buf124
        # Source Nodes: [x_408, x_415, x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 2560, 8, 8), (163840, 64, 8, 1))
        del arg170_1
        del buf125
        buf127 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='cuda', dtype=torch.float32)
        buf128 = reinterpret_tensor(buf127, (8, 2560, 1, 1), (2560, 1, 1, 1), 0); del buf127  # reuse
        # Source Nodes: [x_418, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21.run(buf128, buf126, arg285_1, arg286_1, arg112_1, arg113_1, 20480, 64, grid=grid(20480), stream=stream0)
        del arg112_1
        del arg113_1
        del arg285_1
        del arg286_1
        del buf126
        buf129 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_428], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf128, (8, 2560), (2560, 1), 0), reinterpret_tensor(arg171_1, (2560, 1000), (1, 2560), 0), alpha=1, beta=1, out=buf129)
        del arg171_1
        del arg172_1
        return (buf129, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
