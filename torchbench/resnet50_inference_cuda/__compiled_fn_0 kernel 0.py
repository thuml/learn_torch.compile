
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


# kernel path: /tmp/torchinductor_youkaichao/7e/c7edajh3b7r7vld34sx5nslzqlorfl5flgti2jgo4emdfdd6vcyy.py
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
    size_hints=[16, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2e3bebqzvc7zi4dei55h5bgpn26w7rkjmbs2qkuumsyjqtkjtk.py
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
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/xw/cxw6xsvu2vh7xttqu4srdy2mpfqbb56z74lwb5ydp7ky5ktvbhmc.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfssx52r5dzpzaqk4jrmqof3g7jtrsqkiuxhd6wu2w54lw42snhx.py
# Source Nodes: [identity, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# identity => max_pool2d_with_indices
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x2) + (224*x3) + (12544*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x2) + (224*x3) + (12544*y4)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x2) + (224*x3) + (12544*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x2) + (224*x3) + (12544*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x2) + (224*x3) + (12544*y4)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x2) + (224*x3) + (12544*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x2) + (224*x3) + (12544*y4)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x2) + (224*x3) + (12544*y4)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x2) + (224*x3) + (12544*y4)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (64*x5) + (200704*y1)), tmp69, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrwc7hkm6jmgkh7ehkthrdogrzn4brzjwvurbvrfy4qrr5wdsle.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_1 => add_3, mul_4, mul_5, sub_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2g3224c4cc3j5qunhvq37rfcwm42wqox5kbznyzoxii43ydw27.py
# Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_1 => add_3, mul_4, mul_5, sub_1
# out_2 => relu_1
# out_3 => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': []},
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xsof6wjic2yqjxxc2gvlzvnktirz5jc7k6lnjhehhbnd3bv232.py
# Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_1 => add_9, mul_13, mul_14, sub_4
# identity_2 => relu_3
# out_7 => add_7, mul_10, mul_11, sub_3
# out_8 => add_10
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
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (802816*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cux2l6bkr7tbl7bsrpv5f4lnqueclbfwfstkwteqgk5jhr44pneg.py
# Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_3 => relu_6
# out_17 => add_16, mul_22, mul_23, sub_7
# out_18 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c7/cc7yz6czb2y2cgfyxpiqohbqclz6wsm7imrpgn3skzn62xscr5lk.py
# Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_31 => add_26, mul_34, mul_35, sub_11
# out_32 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ub/cub6ox46s7bbyjhl5ajalsijj26rce73vpaiefxhfh2arc3kdm6l.py
# Source Nodes: [out_31, out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_31 => add_26, mul_34, mul_35, sub_11
# out_32 => relu_10
# out_33 => convolution_12
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc2fbyq4ij2xkyc4pr5jxw2emj6xeqn6qra37uiip3yzjwxu3vm.py
# Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_34 => add_28, mul_37, mul_38, sub_12
# out_35 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbveltk3xiunjphwjaekefnqjpo7z42ug5e6dmtmtrfvvceoczjy.py
# Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_5 => add_32, mul_43, mul_44, sub_14
# identity_6 => relu_12
# out_37 => add_30, mul_40, mul_41, sub_13
# out_38 => add_33
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (512*x2) + (401408*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttckgniqbgqh74tcxtqgmean6u26q3d6qrmpbxrnenxzoews4rr.py
# Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_7 => relu_15
# out_47 => add_39, mul_52, mul_53, sub_17
# out_48 => add_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmknz7kbnezmspvsa6hcfadxsdoonafvqdofosg7x26ea3djgl5.py
# Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_71 => add_56, mul_73, mul_74, sub_24
# out_72 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3m6obzwbldk5bh5mpnft5qrlpmecgcn7kd2yz2b2yoxsgk4s3zb.py
# Source Nodes: [out_71, out_72, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_71 => add_56, mul_73, mul_74, sub_24
# out_72 => relu_22
# out_73 => convolution_25
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtfkkg7xp522ivzg4mmx2atdtms4hdpsra74q6qmbxliqcyzgiu.py
# Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_74 => add_58, mul_76, mul_77, sub_25
# out_75 => relu_23
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (50176*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2krqd4ayernyytwwlsaxwcsiwlf263lkz3dritdhjkpcitaqei.py
# Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_10 => add_62, mul_82, mul_83, sub_27
# identity_11 => relu_24
# out_77 => add_60, mul_79, mul_80, sub_26
# out_78 => add_63
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
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
    tl.store(out_ptr0 + (y0 + (1024*x2) + (200704*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2ekrsxxqxh62ajto4moxbutq4ngsbfjcugfrzprwmfn3gl2tr2c.py
# Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_12 => relu_27
# out_87 => add_69, mul_91, mul_92, sub_30
# out_88 => add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnaiwh6rgtstujxqctlzhdujadc62gfidr5kzygaejkjzgewfy7k.py
# Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_131 => add_100, mul_130, mul_131, sub_43
# out_132 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5tvzgh4murkakylhi7reyfel67hysyb6qk35ys4kfn5tufvss6.py
# Source Nodes: [out_131, out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_131 => add_100, mul_130, mul_131, sub_43
# out_132 => relu_40
# out_133 => convolution_44
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/colhevei34uui6j635faf5kxwsivzuy55adk25oswjnpwltcdzgj.py
# Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_134 => add_102, mul_133, mul_134, sub_44
# out_135 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (512*x2) + (25088*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2j5ukt7c4oluxon6qrjhwirgiz4rpo6zfg44pqvys3w6txaivvl.py
# Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_17 => add_106, mul_139, mul_140, sub_46
# identity_18 => relu_42
# out_137 => add_104, mul_136, mul_137, sub_45
# out_138 => add_107
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (2048*x2) + (100352*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzin56xb4tzgz2in4im42uol56cqopsydsbx4dg777r4q5exes7.py
# Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_19 => relu_45
# out_147 => add_113, mul_148, mul_149, sub_49
# out_148 => add_114
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (2048*y3)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspxddwqyup6mk5nujhqiygdx2xli4f77twbs6wkjyyfkgmdiyvz.py
# Source Nodes: [out_157, out_158, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# out_157 => add_120, mul_157, mul_158, sub_52
# out_158 => add_121
# x_7 => relu_48
# x_8 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
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
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg136_1, (2048, ), (1, ))
    assert_size_stride(arg137_1, (2048, ), (1, ))
    assert_size_stride(arg138_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg160_1, (1000, ), (1, ))
    assert_size_stride(arg161_1, (64, ), (1, ))
    assert_size_stride(arg162_1, (64, ), (1, ))
    assert_size_stride(arg163_1, (), ())
    assert_size_stride(arg164_1, (64, ), (1, ))
    assert_size_stride(arg165_1, (64, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (64, ), (1, ))
    assert_size_stride(arg168_1, (64, ), (1, ))
    assert_size_stride(arg169_1, (), ())
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (64, ), (1, ))
    assert_size_stride(arg177_1, (64, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (64, ), (1, ))
    assert_size_stride(arg180_1, (64, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (64, ), (1, ))
    assert_size_stride(arg186_1, (64, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (64, ), (1, ))
    assert_size_stride(arg189_1, (64, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (128, ), (1, ))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (128, ), (1, ))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (128, ), (1, ))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (128, ), (1, ))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (256, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (256, ), (1, ))
    assert_size_stride(arg249_1, (256, ), (1, ))
    assert_size_stride(arg250_1, (), ())
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (256, ), (1, ))
    assert_size_stride(arg255_1, (256, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (256, ), (1, ))
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (256, ), (1, ))
    assert_size_stride(arg267_1, (256, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (256, ), (1, ))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (256, ), (1, ))
    assert_size_stride(arg285_1, (256, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (2048, ), (1, ))
    assert_size_stride(arg297_1, (2048, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (2048, ), (1, ))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (512, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (2048, ), (1, ))
    assert_size_stride(arg309_1, (2048, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (512, ), (1, ))
    assert_size_stride(arg312_1, (512, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (2048, ), (1, ))
    assert_size_stride(arg318_1, (2048, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg320_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg320_1
        buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, grid=grid(192, 49), stream=stream0)
        del arg0_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 112, 112), (802816, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg161_1, arg162_1, arg1_1, arg2_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg161_1
        del arg162_1
        del arg1_1
        del arg2_1
        buf4 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 256, 3136, grid=grid(256, 3136), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 56, 56), (200704, 3136, 56, 1))
        del arg3_1
        buf6 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, arg164_1, arg165_1, arg4_1, arg5_1, buf6, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg164_1
        del arg165_1
        del arg4_1
        del arg5_1
        del buf5
        buf7 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg6_1, buf7, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 56, 56), (200704, 3136, 56, 1))
        buf9 = buf6; del buf6  # reuse
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, arg167_1, arg168_1, arg7_1, arg8_1, buf9, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg167_1
        del arg168_1
        del arg7_1
        del arg8_1
        del buf8
        # Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del arg9_1
        del buf9
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf4, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del arg12_1
        buf12 = buf10; del buf10  # reuse
        buf13 = reinterpret_tensor(buf3, (4, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf12, arg170_1, arg171_1, arg10_1, arg11_1, buf11, arg173_1, arg174_1, arg13_1, arg14_1, buf13, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg13_1
        del arg14_1
        del arg170_1
        del arg171_1
        del arg173_1
        del arg174_1
        del buf11
        del buf12
        # Source Nodes: [identity_2, out_10], Original ATen: [aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 56, 56), (200704, 3136, 56, 1))
        del arg15_1
        buf15 = buf4; del buf4  # reuse
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf14, arg176_1, arg177_1, arg16_1, arg17_1, buf15, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg16_1
        del arg176_1
        del arg177_1
        del arg17_1
        del buf14
        buf16 = buf7; del buf7  # reuse
        # Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg18_1, buf16, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg18_1
        # Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 64, 56, 56), (200704, 3136, 56, 1))
        buf18 = buf15; del buf15  # reuse
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf17, arg179_1, arg180_1, arg19_1, arg20_1, buf18, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg179_1
        del arg180_1
        del arg19_1
        del arg20_1
        del buf17
        # Source Nodes: [out_14, out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf19 = extern_kernels.convolution(buf18, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del arg21_1
        buf20 = buf13; del buf13  # reuse
        # Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf20, buf19, arg182_1, arg183_1, arg22_1, arg23_1, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del arg182_1
        del arg183_1
        del arg22_1
        del arg23_1
        del buf19
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 64, 56, 56), (200704, 3136, 56, 1))
        del arg24_1
        buf22 = buf18; del buf18  # reuse
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf21, arg185_1, arg186_1, arg25_1, arg26_1, buf22, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg185_1
        del arg186_1
        del arg25_1
        del arg26_1
        del buf21
        buf23 = buf16; del buf16  # reuse
        # Source Nodes: [out_21, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg27_1, buf23, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg27_1
        # Source Nodes: [out_21, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 64, 56, 56), (200704, 3136, 56, 1))
        del buf23
        buf25 = buf22; del buf22  # reuse
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf24, arg188_1, arg189_1, arg28_1, arg29_1, buf25, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg188_1
        del arg189_1
        del arg28_1
        del arg29_1
        del buf24
        # Source Nodes: [out_24, out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del arg30_1
        buf27 = buf20; del buf20  # reuse
        # Source Nodes: [identity_4, out_27, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf27, buf26, arg191_1, arg192_1, arg31_1, arg32_1, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del arg191_1
        del arg192_1
        del arg31_1
        del arg32_1
        del buf26
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg33_1
        buf29 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf28, arg194_1, arg195_1, arg34_1, arg35_1, buf29, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del arg194_1
        del arg195_1
        del arg34_1
        del arg35_1
        del buf28
        buf30 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31, out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg36_1, buf30, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg36_1
        # Source Nodes: [out_31, out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf31 = extern_kernels.convolution(buf29, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf32 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf31, arg197_1, arg198_1, arg37_1, arg38_1, buf32, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg197_1
        del arg198_1
        del arg37_1
        del arg38_1
        del buf31
        # Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf33 = extern_kernels.convolution(buf32, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg39_1
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf27, arg42_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg42_1
        del buf27
        buf35 = buf33; del buf33  # reuse
        buf36 = reinterpret_tensor(buf29, (4, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf29  # reuse
        # Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf35, arg200_1, arg201_1, arg40_1, arg41_1, buf34, arg203_1, arg204_1, arg43_1, arg44_1, buf36, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del arg200_1
        del arg201_1
        del arg203_1
        del arg204_1
        del arg40_1
        del arg41_1
        del arg43_1
        del arg44_1
        del buf34
        del buf35
        # Source Nodes: [identity_6, out_40], Original ATen: [aten.convolution, aten.relu]
        buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg45_1
        buf38 = buf32; del buf32  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf37, arg206_1, arg207_1, arg46_1, arg47_1, buf38, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg206_1
        del arg207_1
        del arg46_1
        del arg47_1
        del buf37
        buf39 = buf30; del buf30  # reuse
        # Source Nodes: [out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg48_1, buf39, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg48_1
        # Source Nodes: [out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf41 = buf38; del buf38  # reuse
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf40, arg209_1, arg210_1, arg49_1, arg50_1, buf41, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg209_1
        del arg210_1
        del arg49_1
        del arg50_1
        del buf40
        # Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg51_1
        buf43 = buf36; del buf36  # reuse
        # Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf43, buf42, arg212_1, arg213_1, arg52_1, arg53_1, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del arg212_1
        del arg213_1
        del arg52_1
        del arg53_1
        del buf42
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg54_1
        buf45 = buf41; del buf41  # reuse
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf44, arg215_1, arg216_1, arg55_1, arg56_1, buf45, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg215_1
        del arg216_1
        del arg55_1
        del arg56_1
        del buf44
        buf46 = buf39; del buf39  # reuse
        # Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg57_1, buf46, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg57_1
        # Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf48 = buf45; del buf45  # reuse
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf47, arg218_1, arg219_1, arg58_1, arg59_1, buf48, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg218_1
        del arg219_1
        del arg58_1
        del arg59_1
        del buf47
        # Source Nodes: [out_54, out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf49 = extern_kernels.convolution(buf48, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg60_1
        buf50 = buf43; del buf43  # reuse
        # Source Nodes: [identity_8, out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf50, buf49, arg221_1, arg222_1, arg61_1, arg62_1, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del arg221_1
        del arg222_1
        del arg61_1
        del arg62_1
        del buf49
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg63_1
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf51, arg224_1, arg225_1, arg64_1, arg65_1, buf52, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg224_1
        del arg225_1
        del arg64_1
        del arg65_1
        del buf51
        buf53 = buf46; del buf46  # reuse
        # Source Nodes: [out_61, out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg66_1, buf53, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg66_1
        # Source Nodes: [out_61, out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 128, 28, 28), (100352, 784, 28, 1))
        del buf53
        buf55 = buf52; del buf52  # reuse
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf54, arg227_1, arg228_1, arg67_1, arg68_1, buf55, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg227_1
        del arg228_1
        del arg67_1
        del arg68_1
        del buf54
        # Source Nodes: [out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg69_1
        buf57 = buf50; del buf50  # reuse
        # Source Nodes: [identity_9, out_67, out_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf57, buf56, arg230_1, arg231_1, arg70_1, arg71_1, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del arg230_1
        del arg231_1
        del arg70_1
        del arg71_1
        del buf56
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 256, 28, 28), (200704, 784, 28, 1))
        del arg72_1
        buf59 = reinterpret_tensor(buf25, (4, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf25  # reuse
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf58, arg233_1, arg234_1, arg73_1, arg74_1, buf59, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg233_1
        del arg234_1
        del arg73_1
        del arg74_1
        del buf58
        buf60 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71, out_72, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg75_1, buf60, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg75_1
        # Source Nodes: [out_71, out_72, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf61 = extern_kernels.convolution(buf59, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 256, 14, 14), (50176, 196, 14, 1))
        buf62 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf61, arg236_1, arg237_1, arg76_1, arg77_1, buf62, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg236_1
        del arg237_1
        del arg76_1
        del arg77_1
        del buf61
        # Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf63 = extern_kernels.convolution(buf62, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg78_1
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf57, arg81_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg81_1
        del buf57
        buf65 = buf63; del buf63  # reuse
        buf66 = reinterpret_tensor(buf59, (4, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf59  # reuse
        # Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf65, arg239_1, arg240_1, arg79_1, arg80_1, buf64, arg242_1, arg243_1, arg82_1, arg83_1, buf66, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del arg239_1
        del arg240_1
        del arg242_1
        del arg243_1
        del arg79_1
        del arg80_1
        del arg82_1
        del arg83_1
        del buf64
        del buf65
        # Source Nodes: [identity_11, out_80], Original ATen: [aten.convolution, aten.relu]
        buf67 = extern_kernels.convolution(buf66, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 256, 14, 14), (50176, 196, 14, 1))
        del arg84_1
        buf68 = buf62; del buf62  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf67, arg245_1, arg246_1, arg85_1, arg86_1, buf68, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg245_1
        del arg246_1
        del arg85_1
        del arg86_1
        del buf67
        buf69 = buf60; del buf60  # reuse
        # Source Nodes: [out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg87_1, buf69, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg87_1
        # Source Nodes: [out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 256, 14, 14), (50176, 196, 14, 1))
        buf71 = buf68; del buf68  # reuse
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf70, arg248_1, arg249_1, arg88_1, arg89_1, buf71, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg248_1
        del arg249_1
        del arg88_1
        del arg89_1
        del buf70
        # Source Nodes: [out_84, out_85, out_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg90_1
        buf73 = buf66; del buf66  # reuse
        # Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf73, buf72, arg251_1, arg252_1, arg91_1, arg92_1, 784, 1024, grid=grid(784, 1024), stream=stream0)
        del arg251_1
        del arg252_1
        del arg91_1
        del arg92_1
        del buf72
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 256, 14, 14), (50176, 196, 14, 1))
        del arg93_1
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf74, arg254_1, arg255_1, arg94_1, arg95_1, buf75, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg254_1
        del arg255_1
        del arg94_1
        del arg95_1
        del buf74
        buf76 = buf69; del buf69  # reuse
        # Source Nodes: [out_91, out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg96_1, buf76, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg96_1
        # Source Nodes: [out_91, out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 256, 14, 14), (50176, 196, 14, 1))
        buf78 = buf75; del buf75  # reuse
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf77, arg257_1, arg258_1, arg97_1, arg98_1, buf78, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg257_1
        del arg258_1
        del arg97_1
        del arg98_1
        del buf77
        # Source Nodes: [out_94, out_95, out_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf79 = extern_kernels.convolution(buf78, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg99_1
        buf80 = buf73; del buf73  # reuse
        # Source Nodes: [identity_13, out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf80, buf79, arg260_1, arg261_1, arg100_1, arg101_1, 784, 1024, grid=grid(784, 1024), stream=stream0)
        del arg100_1
        del arg101_1
        del arg260_1
        del arg261_1
        del buf79
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 256, 14, 14), (50176, 196, 14, 1))
        del arg102_1
        buf82 = buf78; del buf78  # reuse
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf81, arg263_1, arg264_1, arg103_1, arg104_1, buf82, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg103_1
        del arg104_1
        del arg263_1
        del arg264_1
        del buf81
        buf83 = buf76; del buf76  # reuse
        # Source Nodes: [out_101, out_102, out_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg105_1, buf83, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg105_1
        # Source Nodes: [out_101, out_102, out_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 256, 14, 14), (50176, 196, 14, 1))
        buf85 = buf82; del buf82  # reuse
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf84, arg266_1, arg267_1, arg106_1, arg107_1, buf85, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg106_1
        del arg107_1
        del arg266_1
        del arg267_1
        del buf84
        # Source Nodes: [out_104, out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg108_1
        buf87 = buf80; del buf80  # reuse
        # Source Nodes: [identity_14, out_107, out_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf87, buf86, arg269_1, arg270_1, arg109_1, arg110_1, 784, 1024, grid=grid(784, 1024), stream=stream0)
        del arg109_1
        del arg110_1
        del arg269_1
        del arg270_1
        del buf86
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 256, 14, 14), (50176, 196, 14, 1))
        del arg111_1
        buf89 = buf85; del buf85  # reuse
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf88, arg272_1, arg273_1, arg112_1, arg113_1, buf89, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg112_1
        del arg113_1
        del arg272_1
        del arg273_1
        del buf88
        buf90 = buf83; del buf83  # reuse
        # Source Nodes: [out_111, out_112, out_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg114_1, buf90, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg114_1
        # Source Nodes: [out_111, out_112, out_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf91 = extern_kernels.convolution(buf89, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 256, 14, 14), (50176, 196, 14, 1))
        buf92 = buf89; del buf89  # reuse
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf91, arg275_1, arg276_1, arg115_1, arg116_1, buf92, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg115_1
        del arg116_1
        del arg275_1
        del arg276_1
        del buf91
        # Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg117_1
        buf94 = buf87; del buf87  # reuse
        # Source Nodes: [identity_15, out_117, out_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf94, buf93, arg278_1, arg279_1, arg118_1, arg119_1, 784, 1024, grid=grid(784, 1024), stream=stream0)
        del arg118_1
        del arg119_1
        del arg278_1
        del arg279_1
        del buf93
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 256, 14, 14), (50176, 196, 14, 1))
        del arg120_1
        buf96 = buf92; del buf92  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf95, arg281_1, arg282_1, arg121_1, arg122_1, buf96, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg121_1
        del arg122_1
        del arg281_1
        del arg282_1
        del buf95
        buf97 = buf90; del buf90  # reuse
        # Source Nodes: [out_121, out_122, out_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg123_1, buf97, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg123_1
        # Source Nodes: [out_121, out_122, out_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 256, 14, 14), (50176, 196, 14, 1))
        del buf97
        buf99 = buf96; del buf96  # reuse
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf98, arg284_1, arg285_1, arg124_1, arg125_1, buf99, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg124_1
        del arg125_1
        del arg284_1
        del arg285_1
        del buf98
        # Source Nodes: [out_124, out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 1024, 14, 14), (200704, 196, 14, 1))
        del arg126_1
        del buf99
        buf101 = buf94; del buf94  # reuse
        # Source Nodes: [identity_16, out_127, out_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf101, buf100, arg287_1, arg288_1, arg127_1, arg128_1, 784, 1024, grid=grid(784, 1024), stream=stream0)
        del arg127_1
        del arg128_1
        del arg287_1
        del arg288_1
        del buf100
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 512, 14, 14), (100352, 196, 14, 1))
        del arg129_1
        buf103 = reinterpret_tensor(buf55, (4, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf55  # reuse
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf102, arg290_1, arg291_1, arg130_1, arg131_1, buf103, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg130_1
        del arg131_1
        del arg290_1
        del arg291_1
        del buf102
        buf104 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131, out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg132_1, buf104, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg132_1
        # Source Nodes: [out_131, out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 512, 7, 7), (25088, 49, 7, 1))
        buf106 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf105, arg293_1, arg294_1, arg133_1, arg134_1, buf106, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg133_1
        del arg134_1
        del arg293_1
        del arg294_1
        del buf105
        # Source Nodes: [out_134, out_135, out_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf107 = extern_kernels.convolution(buf106, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 2048, 7, 7), (100352, 49, 7, 1))
        del arg135_1
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf101, arg138_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 2048, 7, 7), (100352, 49, 7, 1))
        del arg138_1
        del buf101
        buf109 = buf107; del buf107  # reuse
        buf110 = reinterpret_tensor(buf103, (4, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf103  # reuse
        # Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21.run(buf109, arg296_1, arg297_1, arg136_1, arg137_1, buf108, arg299_1, arg300_1, arg139_1, arg140_1, buf110, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del arg136_1
        del arg137_1
        del arg139_1
        del arg140_1
        del arg296_1
        del arg297_1
        del arg299_1
        del arg300_1
        del buf108
        del buf109
        # Source Nodes: [identity_18, out_140], Original ATen: [aten.convolution, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 512, 7, 7), (25088, 49, 7, 1))
        del arg141_1
        buf112 = buf106; del buf106  # reuse
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf111, arg302_1, arg303_1, arg142_1, arg143_1, buf112, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg142_1
        del arg143_1
        del arg302_1
        del arg303_1
        del buf111
        buf113 = buf104; del buf104  # reuse
        # Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg144_1, buf113, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg144_1
        # Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf114 = extern_kernels.convolution(buf112, buf113, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 512, 7, 7), (25088, 49, 7, 1))
        buf115 = buf112; del buf112  # reuse
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf114, arg305_1, arg306_1, arg145_1, arg146_1, buf115, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg145_1
        del arg146_1
        del arg305_1
        del arg306_1
        del buf114
        # Source Nodes: [out_144, out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf116 = extern_kernels.convolution(buf115, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 2048, 7, 7), (100352, 49, 7, 1))
        del arg147_1
        buf117 = buf110; del buf110  # reuse
        # Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf117, buf116, arg308_1, arg309_1, arg148_1, arg149_1, 196, 2048, grid=grid(196, 2048), stream=stream0)
        del arg148_1
        del arg149_1
        del arg308_1
        del arg309_1
        del buf116
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 512, 7, 7), (25088, 49, 7, 1))
        del arg150_1
        buf119 = buf115; del buf115  # reuse
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf118, arg311_1, arg312_1, arg151_1, arg152_1, buf119, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg151_1
        del arg152_1
        del arg311_1
        del arg312_1
        del buf118
        buf120 = buf113; del buf113  # reuse
        # Source Nodes: [out_151, out_152, out_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg153_1, buf120, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg153_1
        # Source Nodes: [out_151, out_152, out_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf121 = extern_kernels.convolution(buf119, buf120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 512, 7, 7), (25088, 49, 7, 1))
        del buf120
        buf122 = buf119; del buf119  # reuse
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf121, arg314_1, arg315_1, arg154_1, arg155_1, buf122, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg154_1
        del arg155_1
        del arg314_1
        del arg315_1
        del buf121
        # Source Nodes: [out_154, out_155, out_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf123 = extern_kernels.convolution(buf122, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 2048, 7, 7), (100352, 49, 7, 1))
        del arg156_1
        del buf122
        buf124 = empty_strided((4, 2048, 1, 1), (2048, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf125 = reinterpret_tensor(buf124, (4, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf124  # reuse
        # Source Nodes: [out_157, out_158, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf125, buf123, arg317_1, arg318_1, arg157_1, arg158_1, buf117, 8192, 49, grid=grid(8192), stream=stream0)
        del arg157_1
        del arg158_1
        del arg317_1
        del arg318_1
        del buf117
        del buf123
        buf126 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg160_1, reinterpret_tensor(buf125, (4, 2048), (2048, 1), 0), reinterpret_tensor(arg159_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf126)
        del arg159_1
        del arg160_1
        return (buf126, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg164_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg167_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg176_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg179_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg182_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg185_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg188_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg194_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg218_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg224_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg236_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg245_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg248_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg254_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg257_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg266_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg275_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg281_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg296_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg299_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg308_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg311_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg317_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg320_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnet50', benchmark_compiled_module)
