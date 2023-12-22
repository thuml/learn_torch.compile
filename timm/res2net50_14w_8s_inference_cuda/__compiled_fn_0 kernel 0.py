
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
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


# kernel path: /tmp/torchinductor_youkaichao/c2/cc24e62426mxvavozylrfg2vi3xbig7xyrp3lc6v4e5y4zxhfaf2.py
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/25/c256efoixkmaq5hdhm62dhptk6dl6abe4qzid5cv2jq2urrnfjw4.py
# Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# shortcut => max_pool2d_with_indices
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
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzpo6t6lmkclyzztd34ivf32cterxhtx763x6g5yoy2aeb5eylp.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 112
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


# kernel path: /tmp/torchinductor_youkaichao/cu/ccu4i5sunaj7eys4qmfyll6ixtvscojgz2256xgdnyi5hcbxaxpz.py
# Source Nodes: [sp_1], Original ATen: [aten.convolution]
# sp_1 => convolution_2
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpe5qv65ln5gz43gyfkksmly4t3upev6x7miseuvc3yzwegik2m.py
# Source Nodes: [sp_1], Original ATen: [aten.convolution]
# sp_1 => convolution_2
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (126*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpunetaez6dkfjy6eo6szq36gzfnbeo3pufijuhcgwtuv4j7drxv.py
# Source Nodes: [sp_5], Original ATen: [aten.convolution]
# sp_5 => convolution_3
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (43904 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qk6lzbvhm7oo63nvb4p55yd5xzaqav3u43y364bpnl2vugmzzi.py
# Source Nodes: [sp_9], Original ATen: [aten.convolution]
# sp_9 => convolution_4
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (87808 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyscdvhvtd3nlu665s5bjhf7bmt4n63nfvbbj5bi7fg7atjl7b3.py
# Source Nodes: [sp_13], Original ATen: [aten.convolution]
# sp_13 => convolution_5
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw42hi3zmqdl6kmv5nwokoehmll357n3p5k3agdna7lmyrft5eni.py
# Source Nodes: [sp_17], Original ATen: [aten.convolution]
# sp_17 => convolution_6
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (175616 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vffzu27xomahvze2uzxvypx66agkeeqk7eyez2f7ojeotvoixt.py
# Source Nodes: [sp_21], Original ATen: [aten.convolution]
# sp_21 => convolution_7
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (219520 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjyf5v3sflkck4awazrfuqqosa5ifloxtfu64rjdcyzxyhhh3pf.py
# Source Nodes: [sp_25], Original ATen: [aten.convolution]
# sp_25 => convolution_8
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (263424 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwcigezm52uctars5cmcqqver3aepz6thgumfcluewegpfuizfo.py
# Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer1___0___pool => avg_pool2d
triton_poi_fused_avg_pool2d_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 43904)
    x6 = xindex % 43904
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (307271 + x6 + (351232*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (307272 + x6 + (351232*x3)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (307273 + x6 + (351232*x3)), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (307327 + x6 + (351232*x3)), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (307328 + x6 + (351232*x3)), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (307329 + x6 + (351232*x3)), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (307383 + x6 + (351232*x3)), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (307384 + x6 + (351232*x3)), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (307385 + x6 + (351232*x3)), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6 + (351232*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvh5cw6ahzozmw75wytp3uttrnc6y35jtchlnucmal2pil4edndh.py
# Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_2 => add_5, mul_7, mul_8, sub_2
# sp_3 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 14
    x2 = (xindex // 43904)
    x4 = xindex % 43904
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (351232*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmphxulciuc4clh76fjomwvq2nvwp2zoekun3pkpul75f4rnw3pm.py
# Source Nodes: [out_4], Original ATen: [aten.convolution]
# out_4 => convolution_9
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (351232*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5th4v3umx55hheebdysigvhys32x5lcbo4swcypbvwoatvy6ku.py
# Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_5 => add_19, mul_28, mul_29, sub_9
# out_6 => add_22
# shortcut_1 => add_21, mul_31, mul_32, sub_10
# shortcut_2 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
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


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7ftheq4uhco7qosgaeqavfkfmro6pmvbichxbzkdjm4xytjmg3.py
# Source Nodes: [sp_31, sp_32, sp_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_31 => add_26, mul_37, mul_38, sub_12
# sp_32 => relu_11
# sp_33 => add_27
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (43904 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yekt2oxbyvzwruf6hwo4l6pgyemhcmzo7kjyjtbqfcpwbgubup.py
# Source Nodes: [sp_35, sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_35 => add_29, mul_40, mul_41, sub_13
# sp_36 => relu_12
# sp_37 => add_30
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (87808 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cyt3zalkkhroayxiwtgzfq2woawf3rjjnwp6fsk6xtb35txihkiy.py
# Source Nodes: [sp_39, sp_40, sp_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_39 => add_32, mul_43, mul_44, sub_14
# sp_40 => relu_13
# sp_41 => add_33
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (131712 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62jn2zs74qiwoiz6cpzpkagbfidhz2dao3cwniexognp6fiki3e.py
# Source Nodes: [sp_43, sp_44, sp_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_43 => add_35, mul_46, mul_47, sub_15
# sp_44 => relu_14
# sp_45 => add_36
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (175616 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqbsillq2ptwxvsa6gvt2ujn7milcmfuzf6yrffj5qr3xv2oipg.py
# Source Nodes: [sp_47, sp_48, sp_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_47 => add_38, mul_49, mul_50, sub_16
# sp_48 => relu_15
# sp_49 => add_39
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (219520 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cusbj7njv5c3bm6iklhksyrs6tqtl6za3eubwsucygbs2yuqgmvl.py
# Source Nodes: [sp_51, sp_52, sp_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_51 => add_41, mul_52, mul_53, sub_17
# sp_52 => relu_16
# sp_53 => add_42
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (263424 + x2 + (3136*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbabwz42g42cnpjj23r7mkvbvkdkh6q4ucurklhfxlampkiqwbgl.py
# Source Nodes: [cat_30], Original ATen: [aten.cat]
# cat_30 => cat_1
triton_poi_fused_cat_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 43904
    x1 = (xindex // 43904)
    tmp0 = tl.load(in_ptr0 + (307328 + x0 + (351232*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (351232*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65jl6k2pu53ktw3ehlnyovsbyhl4u5vk6xjj2fw7ytg7pao4yyk.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_13 => add_46, mul_58, mul_59, sub_19
# out_14 => add_47
# shortcut_3 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
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


# kernel path: /tmp/torchinductor_youkaichao/we/cwexd63hfelh7n24dl2shzod4d2r62w7jhxwttbvasrmyg5n2ndz.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_25 => add_74, mul_88, mul_89, sub_29
# out_26 => relu_28
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wgkqceidfxxjgsqu5ropm4z2lpxg6nv3hhumhxx2aipu54sq6p.py
# Source Nodes: [sp_88], Original ATen: [aten.convolution]
# sp_88 => convolution_30
triton_poi_fused_convolution_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypi3qmxssxliwum23knhepcp5nur2pnekoc2mymc3ylw57bvnrg.py
# Source Nodes: [sp_88], Original ATen: [aten.convolution]
# sp_88 => convolution_30
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (252*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chveugwncwz3tawj5w6nnsw55qhxvoafb5mhuls5x74kpyhyhv5t.py
# Source Nodes: [sp_92], Original ATen: [aten.convolution]
# sp_92 => convolution_31
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (87808 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hh/chhz6vqyia3fz6nxitmfw3vpy7jpyg4ocuj6gs7xa3uiqewlu3qu.py
# Source Nodes: [sp_96], Original ATen: [aten.convolution]
# sp_96 => convolution_32
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (175616 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bstcjoqgzvoo2fxdob4ud45bcukiudsxkz47luebv6agtr2onx.py
# Source Nodes: [sp_100], Original ATen: [aten.convolution]
# sp_100 => convolution_33
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (263424 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbksv3h7qxwica7w5p2jct63fgwqj4q5tqwwnf4ugn5j5nsviuy3.py
# Source Nodes: [sp_104], Original ATen: [aten.convolution]
# sp_104 => convolution_34
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (351232 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplwez4jaswzlwdy6patv7ueffoh2haoxb34cwt2chfrcuymtzgp.py
# Source Nodes: [sp_108], Original ATen: [aten.convolution]
# sp_108 => convolution_35
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (439040 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfiet63vnoy5fzyuajuteza5e74gomcp7ljecmurmhcx62nw2gg.py
# Source Nodes: [sp_112], Original ATen: [aten.convolution]
# sp_112 => convolution_36
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (526848 + x2 + (3136*y0) + (702464*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chuu63jjaw7myzgt3iaowzr46damj6mhxpho4uby7q3agiyobajo.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 21952)
    x6 = (xindex // 28) % 784
    x7 = xindex % 21952
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (614599 + (2*x0) + (112*x6) + (702464*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (614600 + (2*x0) + (112*x6) + (702464*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (614601 + (2*x0) + (112*x6) + (702464*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (614655 + (2*x0) + (112*x6) + (702464*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (614656 + (2*x0) + (112*x6) + (702464*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (614657 + (2*x0) + (112*x6) + (702464*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (614711 + (2*x0) + (112*x6) + (702464*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (614712 + (2*x0) + (112*x6) + (702464*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (614713 + (2*x0) + (112*x6) + (702464*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (175616*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pxrfgtu5675xy4fbbo6wix4uw2rjwsa7pl45yckjgg2kj4p3rm.py
# Source Nodes: [sp_89, sp_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_89 => add_76, mul_91, mul_92, sub_30
# sp_90 => relu_29
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 28
    x2 = (xindex // 21952)
    x4 = xindex % 21952
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (175616*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/ccl4ub4ym2gh7e3pil63nyazxj6jdyvdn64uiapzzhwnzjqylz7n.py
# Source Nodes: [out_28], Original ATen: [aten.convolution]
# out_28 => convolution_37
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (175616*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6w53mlhoetzqv73it4expw4dnjdaupdrvsq6pxt4fuoobbse5x.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_29 => add_90, mul_112, mul_113, sub_37
# out_30 => add_93
# shortcut_5 => add_92, mul_115, mul_116, sub_38
# shortcut_6 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlielc24b5t2rm3vjg44ym67fufsxv2n5eyvewboizg22t42lob.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_33 => add_95, mul_118, mul_119, sub_39
# out_34 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/le/cletgfiey3danobu2go4cw5gq4ufzwvdqbbb6z52hqdzdom37ray.py
# Source Nodes: [sp_117], Original ATen: [aten.convolution]
# sp_117 => convolution_40
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cac6zhteggtazjrigcrzmkq5gq6yzplk5r6knsgshqqjt7tne3vl.py
# Source Nodes: [sp_118, sp_119, sp_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_118 => add_97, mul_121, mul_122, sub_40
# sp_119 => relu_38
# sp_120 => add_98
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (21952 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgtl3as24sjcak6qmwm5ujhg272sjorklkt2v4s2med5ilja74j.py
# Source Nodes: [sp_122, sp_123, sp_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_122 => add_100, mul_124, mul_125, sub_41
# sp_123 => relu_39
# sp_124 => add_101
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (43904 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfmpaa7zqcrq7vhjja5hvmutuqprrldayvbkcur2zh6tnehszie.py
# Source Nodes: [sp_126, sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_126 => add_103, mul_127, mul_128, sub_42
# sp_127 => relu_40
# sp_128 => add_104
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (65856 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65njsxwcwutqstg7dl4njjuphguavn5fkyi6443dqkgg2dcka3r.py
# Source Nodes: [sp_130, sp_131, sp_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_130 => add_106, mul_130, mul_131, sub_43
# sp_131 => relu_41
# sp_132 => add_107
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (87808 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7kjh4yiaahl3xhirm7h6u3gishhnuxoydup3sqgu4ccu4orcuha.py
# Source Nodes: [sp_134, sp_135, sp_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_134 => add_109, mul_133, mul_134, sub_44
# sp_135 => relu_42
# sp_136 => add_110
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (109760 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbesmi2ld47ifwa4d7fwumkqc4e2gc5mnvecddujlf3k7jochvoc.py
# Source Nodes: [sp_138, sp_139, sp_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_138 => add_112, mul_136, mul_137, sub_45
# sp_139 => relu_43
# sp_140 => add_113
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (131712 + x2 + (784*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7nwnyl6z5q74x3r265toladftqos4gt2kvjvt2o5xoklrzal2wp.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 21952
    x1 = (xindex // 21952)
    tmp0 = tl.load(in_ptr0 + (153664 + x0 + (175616*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (175616*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/carph5q6i4dumd4tn77y2wztvsverya6qdrdcuxkl2tu2e52yg5o.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_37 => add_117, mul_142, mul_143, sub_47
# out_38 => add_118
# shortcut_7 => relu_45
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
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


# kernel path: /tmp/torchinductor_youkaichao/7n/c7njx3cfuivywjv3raiy6gx4d7cgfsicslfkwr7is6ktzsgfy6jk.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_57 => add_170, mul_199, mul_200, sub_66
# out_58 => relu_64
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbiif6hnxxy2qt4hpiw75unnugkex7zu6uc57vgxkrwfgsir4o2f.py
# Source Nodes: [sp_204], Original ATen: [aten.convolution]
# sp_204 => convolution_67
triton_poi_fused_convolution_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqwdlgptzmtm2dlk2yk3n2xh23hymw3n5rnw3jilqzgehzgr7aj.py
# Source Nodes: [sp_204], Original ATen: [aten.convolution]
# sp_204 => convolution_67
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (504*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck25oo6wb3mf3bfhg7hnz4gntr5j25xot4vv6yvksbnv3mf2xtoz.py
# Source Nodes: [sp_208], Original ATen: [aten.convolution]
# sp_208 => convolution_68
triton_poi_fused_convolution_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (43904 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6tq6qgpwzssex4ktkagw2heblbksknvohuve7nptkb3qzulmhf.py
# Source Nodes: [sp_212], Original ATen: [aten.convolution]
# sp_212 => convolution_69
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (87808 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcv26khrj7wv6nzaebzkyyt5jnyrtty2blteziaqqy4lxhnjsfn.py
# Source Nodes: [sp_216], Original ATen: [aten.convolution]
# sp_216 => convolution_70
triton_poi_fused_convolution_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnbfywjv35xkprx7r2eoika46hpf5awex4ecjvtj76gdlosuaif.py
# Source Nodes: [sp_220], Original ATen: [aten.convolution]
# sp_220 => convolution_71
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (175616 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6g5q6qsqjwjxfwbc4fbijbpg42vdk4p4yss4fabq3uyijk5n7o.py
# Source Nodes: [sp_224], Original ATen: [aten.convolution]
# sp_224 => convolution_72
triton_poi_fused_convolution_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (219520 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhnntgibadq3x6xwldjd7ajus65hgkuocvcnewlbpwjl4nzkubx.py
# Source Nodes: [sp_228], Original ATen: [aten.convolution]
# sp_228 => convolution_73
triton_poi_fused_convolution_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (263424 + x2 + (784*y0) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cuf5xbxxczkmczr7zoh52tc4tffxjw623co3okak24ys5cctq4gg.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 10976)
    x6 = (xindex // 14) % 784
    x7 = xindex % 10976
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (307299 + (2*x0) + (56*x6) + (351232*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (307300 + (2*x0) + (56*x6) + (351232*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (307301 + (2*x0) + (56*x6) + (351232*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (307327 + (2*x0) + (56*x6) + (351232*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (307328 + (2*x0) + (56*x6) + (351232*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (307329 + (2*x0) + (56*x6) + (351232*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (307355 + (2*x0) + (56*x6) + (351232*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (307356 + (2*x0) + (56*x6) + (351232*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (307357 + (2*x0) + (56*x6) + (351232*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 29, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (87808*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/ctt4kqmurftbzejlxnqhit66pfmiane5dsmss6wkthtbehdy7jfo.py
# Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_205 => add_172, mul_202, mul_203, sub_67
# sp_206 => relu_65
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 56
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (87808*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czocaovswq2ohm7dwc7iqmpmg5qqoh3d54dhydwdqqea6csootqi.py
# Source Nodes: [out_60], Original ATen: [aten.convolution]
# out_60 => convolution_74
triton_poi_fused_convolution_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3584
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2c2aip6v5bv3oenejrqih6dvet4te5avcpnaspeytghbvgdb2fn.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_61 => add_186, mul_223, mul_224, sub_74
# out_62 => add_189
# shortcut_10 => add_188, mul_226, mul_227, sub_75
# shortcut_11 => relu_72
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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


# kernel path: /tmp/torchinductor_youkaichao/ul/culv47rzfipoxjssw2xriic7dxes34miwgumb2dvag7j2mgbmm2a.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_65 => add_191, mul_229, mul_230, sub_76
# out_66 => relu_73
triton_poi_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqktqzclvluo6vkjv2a3iy2ei72hszgwfs6duqv65v76rzspvih.py
# Source Nodes: [sp_233], Original ATen: [aten.convolution]
# sp_233 => convolution_77
triton_poi_fused_convolution_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (10976*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycaxl3dhsjju5plqp6s2wyzamr6dxtucv5rvmtuootumcg3xdcw.py
# Source Nodes: [sp_234, sp_235, sp_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_234 => add_193, mul_232, mul_233, sub_77
# sp_235 => relu_74
# sp_236 => add_194
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (10976 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqf3q3zgea3a6khooj36yrgbjl44nsd4t23hlef7guc3wcphl7cb.py
# Source Nodes: [sp_238, sp_239, sp_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_238 => add_196, mul_235, mul_236, sub_78
# sp_239 => relu_75
# sp_240 => add_197
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (21952 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq2xyjrpyvpwb3ov2zx726t5gy3kkdn2dnhmv5hd7afjldsprz6.py
# Source Nodes: [sp_242, sp_243, sp_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_242 => add_199, mul_238, mul_239, sub_79
# sp_243 => relu_76
# sp_244 => add_200
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (32928 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7y2e6x4rf4uoxdv3affl5lwtjeamvsqbxs2ftycnhbmsbxbkayq.py
# Source Nodes: [sp_246, sp_247, sp_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_246 => add_202, mul_241, mul_242, sub_80
# sp_247 => relu_77
# sp_248 => add_203
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (43904 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajic2gditpof6y6ls7qzbsflp2ghcakgxx2kf2nbotd3olasdme.py
# Source Nodes: [sp_250, sp_251, sp_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_250 => add_205, mul_244, mul_245, sub_81
# sp_251 => relu_78
# sp_252 => add_206
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (54880 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxo7idwy6zdmyvarcssbvf5nebo6bav2tk7kl6j3e3f3mq27xng.py
# Source Nodes: [sp_254, sp_255, sp_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_254 => add_208, mul_247, mul_248, sub_82
# sp_255 => relu_79
# sp_256 => add_209
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (65856 + x2 + (196*y0) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpobneaops3gtitgvggr6d5zol3ol43xd3u3p2xd4ecffjjnzqra.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat_8
triton_poi_fused_cat_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 10976
    x1 = (xindex // 10976)
    tmp0 = tl.load(in_ptr0 + (76832 + x0 + (87808*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (87808*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxm5ytxbjwe5mrwhepakkd4fuo46vtwjs4l5arsbhjj4zgmsjqw4.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_69 => add_213, mul_253, mul_254, sub_84
# out_70 => add_214
# shortcut_12 => relu_81
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/af/cafxcqzb3z62v2q6lu7ebiqu7fynwuhrwrgdtt4yieojfw6yhd4t.py
# Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_105 => add_316, mul_364, mul_365, sub_121
# out_106 => relu_118
triton_poi_fused__native_batch_norm_legit_no_training_relu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/au/caur6fxye7z5tuvs3cspx2tfn6f4cxnv26f4clc4eieviv57y75q.py
# Source Nodes: [sp_378], Original ATen: [aten.convolution]
# sp_378 => convolution_122
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clb6mwndud7rggvyrbadq2ezczkn4g4ue3ifj4oxmy4myx6ukps3.py
# Source Nodes: [sp_378], Original ATen: [aten.convolution]
# sp_378 => convolution_122
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (1008*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlonawuynrv3p3mn7scllyc4b4vylg4ovgfxd7iz2fx4sg5psma.py
# Source Nodes: [sp_382], Original ATen: [aten.convolution]
# sp_382 => convolution_123
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (21952 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydkbigsennr5jasuxvcuqqhzucj6ugpstad3j7onvgrq46j3fgb.py
# Source Nodes: [sp_386], Original ATen: [aten.convolution]
# sp_386 => convolution_124
triton_poi_fused_convolution_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (43904 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4unsirbrtt7tyijnpnacjy5gl2migftvo6m5idjbibv3acploxl.py
# Source Nodes: [sp_390], Original ATen: [aten.convolution]
# sp_390 => convolution_125
triton_poi_fused_convolution_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (65856 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuobkdamhkk3ajxbx2bulgb7f63s4s4gerg2h5lkr7bg5kn2ol6.py
# Source Nodes: [sp_394], Original ATen: [aten.convolution]
# sp_394 => convolution_126
triton_poi_fused_convolution_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (87808 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxctjtas4ee6gm5lfzqcim4pgnm5f62ocb6zc42l63du3p4zului.py
# Source Nodes: [sp_398], Original ATen: [aten.convolution]
# sp_398 => convolution_127
triton_poi_fused_convolution_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (109760 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjss2mjyf2gpq63n76abz6ty6g7trrrj6qnsh7vyqbcrfyhrir2.py
# Source Nodes: [sp_402], Original ATen: [aten.convolution]
# sp_402 => convolution_128
triton_poi_fused_convolution_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (196*y0) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc56ully3kpe55dd5fazbjtqu2odkz2mjlhfx6dutlpx7mvil25p.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 5488)
    x6 = (xindex // 7) % 784
    x7 = xindex % 5488
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (153649 + (2*x0) + (28*x6) + (175616*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (153650 + (2*x0) + (28*x6) + (175616*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (153651 + (2*x0) + (28*x6) + (175616*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (153663 + (2*x0) + (28*x6) + (175616*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (153664 + (2*x0) + (28*x6) + (175616*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (153665 + (2*x0) + (28*x6) + (175616*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (153677 + (2*x0) + (28*x6) + (175616*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (153678 + (2*x0) + (28*x6) + (175616*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (153679 + (2*x0) + (28*x6) + (175616*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 15, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (43904*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckubzuhxzhxzoac5bqgerzlcibzdwcomhifnvt2kzkmp7c4pczbv.py
# Source Nodes: [sp_379, sp_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_379 => add_318, mul_367, mul_368, sub_122
# sp_380 => relu_119
triton_poi_fused__native_batch_norm_legit_no_training_relu_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 112
    x2 = (xindex // 5488)
    x4 = xindex % 5488
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (43904*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cna7cbu2pirtnw3xcpaxqvesnjreipks73ashoxygkvacsjamrjs.py
# Source Nodes: [out_108], Original ATen: [aten.convolution]
# out_108 => convolution_129
triton_poi_fused_convolution_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 896
    y1 = (yindex // 896)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (896*x2) + (43904*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmgldawdfswklupf2el5yydkwi4g2yxtwkemyntn2qkuj2yigg6.py
# Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_109 => add_332, mul_388, mul_389, sub_129
# out_110 => add_335
# shortcut_17 => add_334, mul_391, mul_392, sub_130
# shortcut_18 => relu_126
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
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


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2pce6dlntlgcxk2w5pesznq5rjcmfgiklcxjgqcbnrmilijlzq.py
# Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_113 => add_337, mul_394, mul_395, sub_131
# out_114 => relu_127
triton_poi_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 896
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbvr27raai2pi4srzhjvdgfez6vh2pkvhhopwmb3h2h36pdnnod.py
# Source Nodes: [sp_407], Original ATen: [aten.convolution]
# sp_407 => convolution_132
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (5488*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutvz774pjzxhh5mpg77vrcysxfybqgf3ziwafdulw62xxqzv5in.py
# Source Nodes: [sp_408, sp_409, sp_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_408 => add_339, mul_397, mul_398, sub_132
# sp_409 => relu_128
# sp_410 => add_340
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (5488 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6ttvzwvmc2o44tqryxet5i52b3gsia24niw4w5c7jjin4bwl5ht.py
# Source Nodes: [sp_412, sp_413, sp_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_412 => add_342, mul_400, mul_401, sub_133
# sp_413 => relu_129
# sp_414 => add_343
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (10976 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cus3iwo6fdcz3htxnkmresw5nhsyyphavr2f47c5nxjibwek2wek.py
# Source Nodes: [sp_416, sp_417, sp_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_416 => add_345, mul_403, mul_404, sub_134
# sp_417 => relu_130
# sp_418 => add_346
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (16464 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqubu4b3c377lyo55ryr6nbueqok7c6q2qfqtvkiozhnf6ftktrv.py
# Source Nodes: [sp_420, sp_421, sp_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_420 => add_348, mul_406, mul_407, sub_135
# sp_421 => relu_131
# sp_422 => add_349
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (21952 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ax/cax3wkxbgrgs6grs3euvvtnhqdmypv4mkunorn4ed67t6wdno3eq.py
# Source Nodes: [sp_424, sp_425, sp_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_424 => add_351, mul_409, mul_410, sub_136
# sp_425 => relu_132
# sp_426 => add_352
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (27440 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwodfjsntimoqry54mpmkccaz3scz24sfb5l4xmc2ap4l5w4g3y.py
# Source Nodes: [sp_428, sp_429, sp_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_428 => add_354, mul_412, mul_413, sub_137
# sp_429 => relu_133
# sp_430 => add_355
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (32928 + x2 + (49*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpie3v75gd4pq325vsrwtiofu6ywg244t36q6l5aukrv2z6wlcb4.py
# Source Nodes: [cat_17], Original ATen: [aten.cat]
# cat_17 => cat_14
triton_poi_fused_cat_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 5488
    x1 = (xindex // 5488)
    tmp0 = tl.load(in_ptr0 + (38416 + x0 + (43904*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (43904*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clr7vamym7h2pgkwpda4dpjfdga4ihivxnwheiv4bwcekjyvno7d.py
# Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_117 => add_359, mul_418, mul_419, sub_139
# out_118 => add_360
# shortcut_19 => relu_135
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_93', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crjcwhi45jgorzyox4omt5jyz5chmu5afntqh7ki4yimqasvbyuj.py
# Source Nodes: [out_125, out_126, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# out_125 => add_384, mul_445, mul_446, sub_148
# out_126 => add_385
# x_8 => relu_144
# x_9 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_94', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (112, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (112, ), (1, ))
    assert_size_stride(arg5_1, (112, ), (1, ))
    assert_size_stride(arg6_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg7_1, (14, ), (1, ))
    assert_size_stride(arg8_1, (14, ), (1, ))
    assert_size_stride(arg9_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg10_1, (14, ), (1, ))
    assert_size_stride(arg11_1, (14, ), (1, ))
    assert_size_stride(arg12_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg13_1, (14, ), (1, ))
    assert_size_stride(arg14_1, (14, ), (1, ))
    assert_size_stride(arg15_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg16_1, (14, ), (1, ))
    assert_size_stride(arg17_1, (14, ), (1, ))
    assert_size_stride(arg18_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg19_1, (14, ), (1, ))
    assert_size_stride(arg20_1, (14, ), (1, ))
    assert_size_stride(arg21_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg22_1, (14, ), (1, ))
    assert_size_stride(arg23_1, (14, ), (1, ))
    assert_size_stride(arg24_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg25_1, (14, ), (1, ))
    assert_size_stride(arg26_1, (14, ), (1, ))
    assert_size_stride(arg27_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg34_1, (112, ), (1, ))
    assert_size_stride(arg35_1, (112, ), (1, ))
    assert_size_stride(arg36_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg37_1, (14, ), (1, ))
    assert_size_stride(arg38_1, (14, ), (1, ))
    assert_size_stride(arg39_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg40_1, (14, ), (1, ))
    assert_size_stride(arg41_1, (14, ), (1, ))
    assert_size_stride(arg42_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg43_1, (14, ), (1, ))
    assert_size_stride(arg44_1, (14, ), (1, ))
    assert_size_stride(arg45_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg46_1, (14, ), (1, ))
    assert_size_stride(arg47_1, (14, ), (1, ))
    assert_size_stride(arg48_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg49_1, (14, ), (1, ))
    assert_size_stride(arg50_1, (14, ), (1, ))
    assert_size_stride(arg51_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg52_1, (14, ), (1, ))
    assert_size_stride(arg53_1, (14, ), (1, ))
    assert_size_stride(arg54_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg55_1, (14, ), (1, ))
    assert_size_stride(arg56_1, (14, ), (1, ))
    assert_size_stride(arg57_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg61_1, (112, ), (1, ))
    assert_size_stride(arg62_1, (112, ), (1, ))
    assert_size_stride(arg63_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg64_1, (14, ), (1, ))
    assert_size_stride(arg65_1, (14, ), (1, ))
    assert_size_stride(arg66_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg67_1, (14, ), (1, ))
    assert_size_stride(arg68_1, (14, ), (1, ))
    assert_size_stride(arg69_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg70_1, (14, ), (1, ))
    assert_size_stride(arg71_1, (14, ), (1, ))
    assert_size_stride(arg72_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg73_1, (14, ), (1, ))
    assert_size_stride(arg74_1, (14, ), (1, ))
    assert_size_stride(arg75_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg76_1, (14, ), (1, ))
    assert_size_stride(arg77_1, (14, ), (1, ))
    assert_size_stride(arg78_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg79_1, (14, ), (1, ))
    assert_size_stride(arg80_1, (14, ), (1, ))
    assert_size_stride(arg81_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg82_1, (14, ), (1, ))
    assert_size_stride(arg83_1, (14, ), (1, ))
    assert_size_stride(arg84_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (224, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg88_1, (224, ), (1, ))
    assert_size_stride(arg89_1, (224, ), (1, ))
    assert_size_stride(arg90_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg91_1, (28, ), (1, ))
    assert_size_stride(arg92_1, (28, ), (1, ))
    assert_size_stride(arg93_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg94_1, (28, ), (1, ))
    assert_size_stride(arg95_1, (28, ), (1, ))
    assert_size_stride(arg96_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg97_1, (28, ), (1, ))
    assert_size_stride(arg98_1, (28, ), (1, ))
    assert_size_stride(arg99_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg100_1, (28, ), (1, ))
    assert_size_stride(arg101_1, (28, ), (1, ))
    assert_size_stride(arg102_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg103_1, (28, ), (1, ))
    assert_size_stride(arg104_1, (28, ), (1, ))
    assert_size_stride(arg105_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg106_1, (28, ), (1, ))
    assert_size_stride(arg107_1, (28, ), (1, ))
    assert_size_stride(arg108_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg109_1, (28, ), (1, ))
    assert_size_stride(arg110_1, (28, ), (1, ))
    assert_size_stride(arg111_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg118_1, (224, ), (1, ))
    assert_size_stride(arg119_1, (224, ), (1, ))
    assert_size_stride(arg120_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg121_1, (28, ), (1, ))
    assert_size_stride(arg122_1, (28, ), (1, ))
    assert_size_stride(arg123_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg124_1, (28, ), (1, ))
    assert_size_stride(arg125_1, (28, ), (1, ))
    assert_size_stride(arg126_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg127_1, (28, ), (1, ))
    assert_size_stride(arg128_1, (28, ), (1, ))
    assert_size_stride(arg129_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg130_1, (28, ), (1, ))
    assert_size_stride(arg131_1, (28, ), (1, ))
    assert_size_stride(arg132_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg133_1, (28, ), (1, ))
    assert_size_stride(arg134_1, (28, ), (1, ))
    assert_size_stride(arg135_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg136_1, (28, ), (1, ))
    assert_size_stride(arg137_1, (28, ), (1, ))
    assert_size_stride(arg138_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg139_1, (28, ), (1, ))
    assert_size_stride(arg140_1, (28, ), (1, ))
    assert_size_stride(arg141_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg145_1, (224, ), (1, ))
    assert_size_stride(arg146_1, (224, ), (1, ))
    assert_size_stride(arg147_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg148_1, (28, ), (1, ))
    assert_size_stride(arg149_1, (28, ), (1, ))
    assert_size_stride(arg150_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg151_1, (28, ), (1, ))
    assert_size_stride(arg152_1, (28, ), (1, ))
    assert_size_stride(arg153_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg154_1, (28, ), (1, ))
    assert_size_stride(arg155_1, (28, ), (1, ))
    assert_size_stride(arg156_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg157_1, (28, ), (1, ))
    assert_size_stride(arg158_1, (28, ), (1, ))
    assert_size_stride(arg159_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg160_1, (28, ), (1, ))
    assert_size_stride(arg161_1, (28, ), (1, ))
    assert_size_stride(arg162_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg163_1, (28, ), (1, ))
    assert_size_stride(arg164_1, (28, ), (1, ))
    assert_size_stride(arg165_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg166_1, (28, ), (1, ))
    assert_size_stride(arg167_1, (28, ), (1, ))
    assert_size_stride(arg168_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg169_1, (512, ), (1, ))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg172_1, (224, ), (1, ))
    assert_size_stride(arg173_1, (224, ), (1, ))
    assert_size_stride(arg174_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg175_1, (28, ), (1, ))
    assert_size_stride(arg176_1, (28, ), (1, ))
    assert_size_stride(arg177_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg178_1, (28, ), (1, ))
    assert_size_stride(arg179_1, (28, ), (1, ))
    assert_size_stride(arg180_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg181_1, (28, ), (1, ))
    assert_size_stride(arg182_1, (28, ), (1, ))
    assert_size_stride(arg183_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg184_1, (28, ), (1, ))
    assert_size_stride(arg185_1, (28, ), (1, ))
    assert_size_stride(arg186_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg187_1, (28, ), (1, ))
    assert_size_stride(arg188_1, (28, ), (1, ))
    assert_size_stride(arg189_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg190_1, (28, ), (1, ))
    assert_size_stride(arg191_1, (28, ), (1, ))
    assert_size_stride(arg192_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg193_1, (28, ), (1, ))
    assert_size_stride(arg194_1, (28, ), (1, ))
    assert_size_stride(arg195_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, ), (1, ))
    assert_size_stride(arg198_1, (448, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg199_1, (448, ), (1, ))
    assert_size_stride(arg200_1, (448, ), (1, ))
    assert_size_stride(arg201_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg202_1, (56, ), (1, ))
    assert_size_stride(arg203_1, (56, ), (1, ))
    assert_size_stride(arg204_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg205_1, (56, ), (1, ))
    assert_size_stride(arg206_1, (56, ), (1, ))
    assert_size_stride(arg207_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg208_1, (56, ), (1, ))
    assert_size_stride(arg209_1, (56, ), (1, ))
    assert_size_stride(arg210_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg211_1, (56, ), (1, ))
    assert_size_stride(arg212_1, (56, ), (1, ))
    assert_size_stride(arg213_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg214_1, (56, ), (1, ))
    assert_size_stride(arg215_1, (56, ), (1, ))
    assert_size_stride(arg216_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg217_1, (56, ), (1, ))
    assert_size_stride(arg218_1, (56, ), (1, ))
    assert_size_stride(arg219_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg220_1, (56, ), (1, ))
    assert_size_stride(arg221_1, (56, ), (1, ))
    assert_size_stride(arg222_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg229_1, (448, ), (1, ))
    assert_size_stride(arg230_1, (448, ), (1, ))
    assert_size_stride(arg231_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg232_1, (56, ), (1, ))
    assert_size_stride(arg233_1, (56, ), (1, ))
    assert_size_stride(arg234_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg235_1, (56, ), (1, ))
    assert_size_stride(arg236_1, (56, ), (1, ))
    assert_size_stride(arg237_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg238_1, (56, ), (1, ))
    assert_size_stride(arg239_1, (56, ), (1, ))
    assert_size_stride(arg240_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg241_1, (56, ), (1, ))
    assert_size_stride(arg242_1, (56, ), (1, ))
    assert_size_stride(arg243_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg244_1, (56, ), (1, ))
    assert_size_stride(arg245_1, (56, ), (1, ))
    assert_size_stride(arg246_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg247_1, (56, ), (1, ))
    assert_size_stride(arg248_1, (56, ), (1, ))
    assert_size_stride(arg249_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg250_1, (56, ), (1, ))
    assert_size_stride(arg251_1, (56, ), (1, ))
    assert_size_stride(arg252_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg256_1, (448, ), (1, ))
    assert_size_stride(arg257_1, (448, ), (1, ))
    assert_size_stride(arg258_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg259_1, (56, ), (1, ))
    assert_size_stride(arg260_1, (56, ), (1, ))
    assert_size_stride(arg261_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg262_1, (56, ), (1, ))
    assert_size_stride(arg263_1, (56, ), (1, ))
    assert_size_stride(arg264_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg265_1, (56, ), (1, ))
    assert_size_stride(arg266_1, (56, ), (1, ))
    assert_size_stride(arg267_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg268_1, (56, ), (1, ))
    assert_size_stride(arg269_1, (56, ), (1, ))
    assert_size_stride(arg270_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg271_1, (56, ), (1, ))
    assert_size_stride(arg272_1, (56, ), (1, ))
    assert_size_stride(arg273_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg274_1, (56, ), (1, ))
    assert_size_stride(arg275_1, (56, ), (1, ))
    assert_size_stride(arg276_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg277_1, (56, ), (1, ))
    assert_size_stride(arg278_1, (56, ), (1, ))
    assert_size_stride(arg279_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg283_1, (448, ), (1, ))
    assert_size_stride(arg284_1, (448, ), (1, ))
    assert_size_stride(arg285_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg286_1, (56, ), (1, ))
    assert_size_stride(arg287_1, (56, ), (1, ))
    assert_size_stride(arg288_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg289_1, (56, ), (1, ))
    assert_size_stride(arg290_1, (56, ), (1, ))
    assert_size_stride(arg291_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg292_1, (56, ), (1, ))
    assert_size_stride(arg293_1, (56, ), (1, ))
    assert_size_stride(arg294_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg295_1, (56, ), (1, ))
    assert_size_stride(arg296_1, (56, ), (1, ))
    assert_size_stride(arg297_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg298_1, (56, ), (1, ))
    assert_size_stride(arg299_1, (56, ), (1, ))
    assert_size_stride(arg300_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg301_1, (56, ), (1, ))
    assert_size_stride(arg302_1, (56, ), (1, ))
    assert_size_stride(arg303_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg304_1, (56, ), (1, ))
    assert_size_stride(arg305_1, (56, ), (1, ))
    assert_size_stride(arg306_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg310_1, (448, ), (1, ))
    assert_size_stride(arg311_1, (448, ), (1, ))
    assert_size_stride(arg312_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg313_1, (56, ), (1, ))
    assert_size_stride(arg314_1, (56, ), (1, ))
    assert_size_stride(arg315_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg316_1, (56, ), (1, ))
    assert_size_stride(arg317_1, (56, ), (1, ))
    assert_size_stride(arg318_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg319_1, (56, ), (1, ))
    assert_size_stride(arg320_1, (56, ), (1, ))
    assert_size_stride(arg321_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg322_1, (56, ), (1, ))
    assert_size_stride(arg323_1, (56, ), (1, ))
    assert_size_stride(arg324_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg325_1, (56, ), (1, ))
    assert_size_stride(arg326_1, (56, ), (1, ))
    assert_size_stride(arg327_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg328_1, (56, ), (1, ))
    assert_size_stride(arg329_1, (56, ), (1, ))
    assert_size_stride(arg330_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg331_1, (56, ), (1, ))
    assert_size_stride(arg332_1, (56, ), (1, ))
    assert_size_stride(arg333_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg337_1, (448, ), (1, ))
    assert_size_stride(arg338_1, (448, ), (1, ))
    assert_size_stride(arg339_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg340_1, (56, ), (1, ))
    assert_size_stride(arg341_1, (56, ), (1, ))
    assert_size_stride(arg342_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg343_1, (56, ), (1, ))
    assert_size_stride(arg344_1, (56, ), (1, ))
    assert_size_stride(arg345_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg346_1, (56, ), (1, ))
    assert_size_stride(arg347_1, (56, ), (1, ))
    assert_size_stride(arg348_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg349_1, (56, ), (1, ))
    assert_size_stride(arg350_1, (56, ), (1, ))
    assert_size_stride(arg351_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg352_1, (56, ), (1, ))
    assert_size_stride(arg353_1, (56, ), (1, ))
    assert_size_stride(arg354_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg355_1, (56, ), (1, ))
    assert_size_stride(arg356_1, (56, ), (1, ))
    assert_size_stride(arg357_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg358_1, (56, ), (1, ))
    assert_size_stride(arg359_1, (56, ), (1, ))
    assert_size_stride(arg360_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (896, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg364_1, (896, ), (1, ))
    assert_size_stride(arg365_1, (896, ), (1, ))
    assert_size_stride(arg366_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg367_1, (112, ), (1, ))
    assert_size_stride(arg368_1, (112, ), (1, ))
    assert_size_stride(arg369_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg370_1, (112, ), (1, ))
    assert_size_stride(arg371_1, (112, ), (1, ))
    assert_size_stride(arg372_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg373_1, (112, ), (1, ))
    assert_size_stride(arg374_1, (112, ), (1, ))
    assert_size_stride(arg375_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg376_1, (112, ), (1, ))
    assert_size_stride(arg377_1, (112, ), (1, ))
    assert_size_stride(arg378_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg379_1, (112, ), (1, ))
    assert_size_stride(arg380_1, (112, ), (1, ))
    assert_size_stride(arg381_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg382_1, (112, ), (1, ))
    assert_size_stride(arg383_1, (112, ), (1, ))
    assert_size_stride(arg384_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg385_1, (112, ), (1, ))
    assert_size_stride(arg386_1, (112, ), (1, ))
    assert_size_stride(arg387_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg388_1, (2048, ), (1, ))
    assert_size_stride(arg389_1, (2048, ), (1, ))
    assert_size_stride(arg390_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg391_1, (2048, ), (1, ))
    assert_size_stride(arg392_1, (2048, ), (1, ))
    assert_size_stride(arg393_1, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg394_1, (896, ), (1, ))
    assert_size_stride(arg395_1, (896, ), (1, ))
    assert_size_stride(arg396_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg397_1, (112, ), (1, ))
    assert_size_stride(arg398_1, (112, ), (1, ))
    assert_size_stride(arg399_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg400_1, (112, ), (1, ))
    assert_size_stride(arg401_1, (112, ), (1, ))
    assert_size_stride(arg402_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg403_1, (112, ), (1, ))
    assert_size_stride(arg404_1, (112, ), (1, ))
    assert_size_stride(arg405_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg406_1, (112, ), (1, ))
    assert_size_stride(arg407_1, (112, ), (1, ))
    assert_size_stride(arg408_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg409_1, (112, ), (1, ))
    assert_size_stride(arg410_1, (112, ), (1, ))
    assert_size_stride(arg411_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg412_1, (112, ), (1, ))
    assert_size_stride(arg413_1, (112, ), (1, ))
    assert_size_stride(arg414_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg415_1, (112, ), (1, ))
    assert_size_stride(arg416_1, (112, ), (1, ))
    assert_size_stride(arg417_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg418_1, (2048, ), (1, ))
    assert_size_stride(arg419_1, (2048, ), (1, ))
    assert_size_stride(arg420_1, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg421_1, (896, ), (1, ))
    assert_size_stride(arg422_1, (896, ), (1, ))
    assert_size_stride(arg423_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg424_1, (112, ), (1, ))
    assert_size_stride(arg425_1, (112, ), (1, ))
    assert_size_stride(arg426_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg427_1, (112, ), (1, ))
    assert_size_stride(arg428_1, (112, ), (1, ))
    assert_size_stride(arg429_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg430_1, (112, ), (1, ))
    assert_size_stride(arg431_1, (112, ), (1, ))
    assert_size_stride(arg432_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg433_1, (112, ), (1, ))
    assert_size_stride(arg434_1, (112, ), (1, ))
    assert_size_stride(arg435_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg436_1, (112, ), (1, ))
    assert_size_stride(arg437_1, (112, ), (1, ))
    assert_size_stride(arg438_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg439_1, (112, ), (1, ))
    assert_size_stride(arg440_1, (112, ), (1, ))
    assert_size_stride(arg441_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg442_1, (112, ), (1, ))
    assert_size_stride(arg443_1, (112, ), (1, ))
    assert_size_stride(arg444_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg445_1, (2048, ), (1, ))
    assert_size_stride(arg446_1, (2048, ), (1, ))
    assert_size_stride(arg447_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg448_1, (1000, ), (1, ))
    assert_size_stride(arg449_1, (64, ), (1, ))
    assert_size_stride(arg450_1, (64, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (112, ), (1, ))
    assert_size_stride(arg453_1, (112, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (14, ), (1, ))
    assert_size_stride(arg456_1, (14, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (14, ), (1, ))
    assert_size_stride(arg459_1, (14, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (14, ), (1, ))
    assert_size_stride(arg462_1, (14, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (14, ), (1, ))
    assert_size_stride(arg465_1, (14, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (14, ), (1, ))
    assert_size_stride(arg468_1, (14, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (14, ), (1, ))
    assert_size_stride(arg471_1, (14, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (14, ), (1, ))
    assert_size_stride(arg474_1, (14, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (256, ), (1, ))
    assert_size_stride(arg477_1, (256, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (256, ), (1, ))
    assert_size_stride(arg480_1, (256, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (112, ), (1, ))
    assert_size_stride(arg483_1, (112, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (14, ), (1, ))
    assert_size_stride(arg486_1, (14, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (14, ), (1, ))
    assert_size_stride(arg489_1, (14, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (14, ), (1, ))
    assert_size_stride(arg492_1, (14, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (14, ), (1, ))
    assert_size_stride(arg495_1, (14, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (14, ), (1, ))
    assert_size_stride(arg498_1, (14, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (14, ), (1, ))
    assert_size_stride(arg501_1, (14, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (14, ), (1, ))
    assert_size_stride(arg504_1, (14, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (256, ), (1, ))
    assert_size_stride(arg507_1, (256, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (112, ), (1, ))
    assert_size_stride(arg510_1, (112, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (14, ), (1, ))
    assert_size_stride(arg513_1, (14, ), (1, ))
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (14, ), (1, ))
    assert_size_stride(arg516_1, (14, ), (1, ))
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (14, ), (1, ))
    assert_size_stride(arg519_1, (14, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (14, ), (1, ))
    assert_size_stride(arg522_1, (14, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (14, ), (1, ))
    assert_size_stride(arg525_1, (14, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (14, ), (1, ))
    assert_size_stride(arg528_1, (14, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (14, ), (1, ))
    assert_size_stride(arg531_1, (14, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (256, ), (1, ))
    assert_size_stride(arg534_1, (256, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (224, ), (1, ))
    assert_size_stride(arg537_1, (224, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (28, ), (1, ))
    assert_size_stride(arg540_1, (28, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (28, ), (1, ))
    assert_size_stride(arg543_1, (28, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (28, ), (1, ))
    assert_size_stride(arg546_1, (28, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (28, ), (1, ))
    assert_size_stride(arg549_1, (28, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (28, ), (1, ))
    assert_size_stride(arg552_1, (28, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (28, ), (1, ))
    assert_size_stride(arg555_1, (28, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (28, ), (1, ))
    assert_size_stride(arg558_1, (28, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (512, ), (1, ))
    assert_size_stride(arg561_1, (512, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (512, ), (1, ))
    assert_size_stride(arg564_1, (512, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (224, ), (1, ))
    assert_size_stride(arg567_1, (224, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (28, ), (1, ))
    assert_size_stride(arg570_1, (28, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (28, ), (1, ))
    assert_size_stride(arg573_1, (28, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (28, ), (1, ))
    assert_size_stride(arg576_1, (28, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (28, ), (1, ))
    assert_size_stride(arg579_1, (28, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (28, ), (1, ))
    assert_size_stride(arg582_1, (28, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (28, ), (1, ))
    assert_size_stride(arg585_1, (28, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (28, ), (1, ))
    assert_size_stride(arg588_1, (28, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (512, ), (1, ))
    assert_size_stride(arg591_1, (512, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (224, ), (1, ))
    assert_size_stride(arg594_1, (224, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (28, ), (1, ))
    assert_size_stride(arg597_1, (28, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (28, ), (1, ))
    assert_size_stride(arg600_1, (28, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (28, ), (1, ))
    assert_size_stride(arg603_1, (28, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (28, ), (1, ))
    assert_size_stride(arg606_1, (28, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (28, ), (1, ))
    assert_size_stride(arg609_1, (28, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (28, ), (1, ))
    assert_size_stride(arg612_1, (28, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (28, ), (1, ))
    assert_size_stride(arg615_1, (28, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (512, ), (1, ))
    assert_size_stride(arg618_1, (512, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (224, ), (1, ))
    assert_size_stride(arg621_1, (224, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (28, ), (1, ))
    assert_size_stride(arg624_1, (28, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (28, ), (1, ))
    assert_size_stride(arg627_1, (28, ), (1, ))
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (28, ), (1, ))
    assert_size_stride(arg630_1, (28, ), (1, ))
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (28, ), (1, ))
    assert_size_stride(arg633_1, (28, ), (1, ))
    assert_size_stride(arg634_1, (), ())
    assert_size_stride(arg635_1, (28, ), (1, ))
    assert_size_stride(arg636_1, (28, ), (1, ))
    assert_size_stride(arg637_1, (), ())
    assert_size_stride(arg638_1, (28, ), (1, ))
    assert_size_stride(arg639_1, (28, ), (1, ))
    assert_size_stride(arg640_1, (), ())
    assert_size_stride(arg641_1, (28, ), (1, ))
    assert_size_stride(arg642_1, (28, ), (1, ))
    assert_size_stride(arg643_1, (), ())
    assert_size_stride(arg644_1, (512, ), (1, ))
    assert_size_stride(arg645_1, (512, ), (1, ))
    assert_size_stride(arg646_1, (), ())
    assert_size_stride(arg647_1, (448, ), (1, ))
    assert_size_stride(arg648_1, (448, ), (1, ))
    assert_size_stride(arg649_1, (), ())
    assert_size_stride(arg650_1, (56, ), (1, ))
    assert_size_stride(arg651_1, (56, ), (1, ))
    assert_size_stride(arg652_1, (), ())
    assert_size_stride(arg653_1, (56, ), (1, ))
    assert_size_stride(arg654_1, (56, ), (1, ))
    assert_size_stride(arg655_1, (), ())
    assert_size_stride(arg656_1, (56, ), (1, ))
    assert_size_stride(arg657_1, (56, ), (1, ))
    assert_size_stride(arg658_1, (), ())
    assert_size_stride(arg659_1, (56, ), (1, ))
    assert_size_stride(arg660_1, (56, ), (1, ))
    assert_size_stride(arg661_1, (), ())
    assert_size_stride(arg662_1, (56, ), (1, ))
    assert_size_stride(arg663_1, (56, ), (1, ))
    assert_size_stride(arg664_1, (), ())
    assert_size_stride(arg665_1, (56, ), (1, ))
    assert_size_stride(arg666_1, (56, ), (1, ))
    assert_size_stride(arg667_1, (), ())
    assert_size_stride(arg668_1, (56, ), (1, ))
    assert_size_stride(arg669_1, (56, ), (1, ))
    assert_size_stride(arg670_1, (), ())
    assert_size_stride(arg671_1, (1024, ), (1, ))
    assert_size_stride(arg672_1, (1024, ), (1, ))
    assert_size_stride(arg673_1, (), ())
    assert_size_stride(arg674_1, (1024, ), (1, ))
    assert_size_stride(arg675_1, (1024, ), (1, ))
    assert_size_stride(arg676_1, (), ())
    assert_size_stride(arg677_1, (448, ), (1, ))
    assert_size_stride(arg678_1, (448, ), (1, ))
    assert_size_stride(arg679_1, (), ())
    assert_size_stride(arg680_1, (56, ), (1, ))
    assert_size_stride(arg681_1, (56, ), (1, ))
    assert_size_stride(arg682_1, (), ())
    assert_size_stride(arg683_1, (56, ), (1, ))
    assert_size_stride(arg684_1, (56, ), (1, ))
    assert_size_stride(arg685_1, (), ())
    assert_size_stride(arg686_1, (56, ), (1, ))
    assert_size_stride(arg687_1, (56, ), (1, ))
    assert_size_stride(arg688_1, (), ())
    assert_size_stride(arg689_1, (56, ), (1, ))
    assert_size_stride(arg690_1, (56, ), (1, ))
    assert_size_stride(arg691_1, (), ())
    assert_size_stride(arg692_1, (56, ), (1, ))
    assert_size_stride(arg693_1, (56, ), (1, ))
    assert_size_stride(arg694_1, (), ())
    assert_size_stride(arg695_1, (56, ), (1, ))
    assert_size_stride(arg696_1, (56, ), (1, ))
    assert_size_stride(arg697_1, (), ())
    assert_size_stride(arg698_1, (56, ), (1, ))
    assert_size_stride(arg699_1, (56, ), (1, ))
    assert_size_stride(arg700_1, (), ())
    assert_size_stride(arg701_1, (1024, ), (1, ))
    assert_size_stride(arg702_1, (1024, ), (1, ))
    assert_size_stride(arg703_1, (), ())
    assert_size_stride(arg704_1, (448, ), (1, ))
    assert_size_stride(arg705_1, (448, ), (1, ))
    assert_size_stride(arg706_1, (), ())
    assert_size_stride(arg707_1, (56, ), (1, ))
    assert_size_stride(arg708_1, (56, ), (1, ))
    assert_size_stride(arg709_1, (), ())
    assert_size_stride(arg710_1, (56, ), (1, ))
    assert_size_stride(arg711_1, (56, ), (1, ))
    assert_size_stride(arg712_1, (), ())
    assert_size_stride(arg713_1, (56, ), (1, ))
    assert_size_stride(arg714_1, (56, ), (1, ))
    assert_size_stride(arg715_1, (), ())
    assert_size_stride(arg716_1, (56, ), (1, ))
    assert_size_stride(arg717_1, (56, ), (1, ))
    assert_size_stride(arg718_1, (), ())
    assert_size_stride(arg719_1, (56, ), (1, ))
    assert_size_stride(arg720_1, (56, ), (1, ))
    assert_size_stride(arg721_1, (), ())
    assert_size_stride(arg722_1, (56, ), (1, ))
    assert_size_stride(arg723_1, (56, ), (1, ))
    assert_size_stride(arg724_1, (), ())
    assert_size_stride(arg725_1, (56, ), (1, ))
    assert_size_stride(arg726_1, (56, ), (1, ))
    assert_size_stride(arg727_1, (), ())
    assert_size_stride(arg728_1, (1024, ), (1, ))
    assert_size_stride(arg729_1, (1024, ), (1, ))
    assert_size_stride(arg730_1, (), ())
    assert_size_stride(arg731_1, (448, ), (1, ))
    assert_size_stride(arg732_1, (448, ), (1, ))
    assert_size_stride(arg733_1, (), ())
    assert_size_stride(arg734_1, (56, ), (1, ))
    assert_size_stride(arg735_1, (56, ), (1, ))
    assert_size_stride(arg736_1, (), ())
    assert_size_stride(arg737_1, (56, ), (1, ))
    assert_size_stride(arg738_1, (56, ), (1, ))
    assert_size_stride(arg739_1, (), ())
    assert_size_stride(arg740_1, (56, ), (1, ))
    assert_size_stride(arg741_1, (56, ), (1, ))
    assert_size_stride(arg742_1, (), ())
    assert_size_stride(arg743_1, (56, ), (1, ))
    assert_size_stride(arg744_1, (56, ), (1, ))
    assert_size_stride(arg745_1, (), ())
    assert_size_stride(arg746_1, (56, ), (1, ))
    assert_size_stride(arg747_1, (56, ), (1, ))
    assert_size_stride(arg748_1, (), ())
    assert_size_stride(arg749_1, (56, ), (1, ))
    assert_size_stride(arg750_1, (56, ), (1, ))
    assert_size_stride(arg751_1, (), ())
    assert_size_stride(arg752_1, (56, ), (1, ))
    assert_size_stride(arg753_1, (56, ), (1, ))
    assert_size_stride(arg754_1, (), ())
    assert_size_stride(arg755_1, (1024, ), (1, ))
    assert_size_stride(arg756_1, (1024, ), (1, ))
    assert_size_stride(arg757_1, (), ())
    assert_size_stride(arg758_1, (448, ), (1, ))
    assert_size_stride(arg759_1, (448, ), (1, ))
    assert_size_stride(arg760_1, (), ())
    assert_size_stride(arg761_1, (56, ), (1, ))
    assert_size_stride(arg762_1, (56, ), (1, ))
    assert_size_stride(arg763_1, (), ())
    assert_size_stride(arg764_1, (56, ), (1, ))
    assert_size_stride(arg765_1, (56, ), (1, ))
    assert_size_stride(arg766_1, (), ())
    assert_size_stride(arg767_1, (56, ), (1, ))
    assert_size_stride(arg768_1, (56, ), (1, ))
    assert_size_stride(arg769_1, (), ())
    assert_size_stride(arg770_1, (56, ), (1, ))
    assert_size_stride(arg771_1, (56, ), (1, ))
    assert_size_stride(arg772_1, (), ())
    assert_size_stride(arg773_1, (56, ), (1, ))
    assert_size_stride(arg774_1, (56, ), (1, ))
    assert_size_stride(arg775_1, (), ())
    assert_size_stride(arg776_1, (56, ), (1, ))
    assert_size_stride(arg777_1, (56, ), (1, ))
    assert_size_stride(arg778_1, (), ())
    assert_size_stride(arg779_1, (56, ), (1, ))
    assert_size_stride(arg780_1, (56, ), (1, ))
    assert_size_stride(arg781_1, (), ())
    assert_size_stride(arg782_1, (1024, ), (1, ))
    assert_size_stride(arg783_1, (1024, ), (1, ))
    assert_size_stride(arg784_1, (), ())
    assert_size_stride(arg785_1, (448, ), (1, ))
    assert_size_stride(arg786_1, (448, ), (1, ))
    assert_size_stride(arg787_1, (), ())
    assert_size_stride(arg788_1, (56, ), (1, ))
    assert_size_stride(arg789_1, (56, ), (1, ))
    assert_size_stride(arg790_1, (), ())
    assert_size_stride(arg791_1, (56, ), (1, ))
    assert_size_stride(arg792_1, (56, ), (1, ))
    assert_size_stride(arg793_1, (), ())
    assert_size_stride(arg794_1, (56, ), (1, ))
    assert_size_stride(arg795_1, (56, ), (1, ))
    assert_size_stride(arg796_1, (), ())
    assert_size_stride(arg797_1, (56, ), (1, ))
    assert_size_stride(arg798_1, (56, ), (1, ))
    assert_size_stride(arg799_1, (), ())
    assert_size_stride(arg800_1, (56, ), (1, ))
    assert_size_stride(arg801_1, (56, ), (1, ))
    assert_size_stride(arg802_1, (), ())
    assert_size_stride(arg803_1, (56, ), (1, ))
    assert_size_stride(arg804_1, (56, ), (1, ))
    assert_size_stride(arg805_1, (), ())
    assert_size_stride(arg806_1, (56, ), (1, ))
    assert_size_stride(arg807_1, (56, ), (1, ))
    assert_size_stride(arg808_1, (), ())
    assert_size_stride(arg809_1, (1024, ), (1, ))
    assert_size_stride(arg810_1, (1024, ), (1, ))
    assert_size_stride(arg811_1, (), ())
    assert_size_stride(arg812_1, (896, ), (1, ))
    assert_size_stride(arg813_1, (896, ), (1, ))
    assert_size_stride(arg814_1, (), ())
    assert_size_stride(arg815_1, (112, ), (1, ))
    assert_size_stride(arg816_1, (112, ), (1, ))
    assert_size_stride(arg817_1, (), ())
    assert_size_stride(arg818_1, (112, ), (1, ))
    assert_size_stride(arg819_1, (112, ), (1, ))
    assert_size_stride(arg820_1, (), ())
    assert_size_stride(arg821_1, (112, ), (1, ))
    assert_size_stride(arg822_1, (112, ), (1, ))
    assert_size_stride(arg823_1, (), ())
    assert_size_stride(arg824_1, (112, ), (1, ))
    assert_size_stride(arg825_1, (112, ), (1, ))
    assert_size_stride(arg826_1, (), ())
    assert_size_stride(arg827_1, (112, ), (1, ))
    assert_size_stride(arg828_1, (112, ), (1, ))
    assert_size_stride(arg829_1, (), ())
    assert_size_stride(arg830_1, (112, ), (1, ))
    assert_size_stride(arg831_1, (112, ), (1, ))
    assert_size_stride(arg832_1, (), ())
    assert_size_stride(arg833_1, (112, ), (1, ))
    assert_size_stride(arg834_1, (112, ), (1, ))
    assert_size_stride(arg835_1, (), ())
    assert_size_stride(arg836_1, (2048, ), (1, ))
    assert_size_stride(arg837_1, (2048, ), (1, ))
    assert_size_stride(arg838_1, (), ())
    assert_size_stride(arg839_1, (2048, ), (1, ))
    assert_size_stride(arg840_1, (2048, ), (1, ))
    assert_size_stride(arg841_1, (), ())
    assert_size_stride(arg842_1, (896, ), (1, ))
    assert_size_stride(arg843_1, (896, ), (1, ))
    assert_size_stride(arg844_1, (), ())
    assert_size_stride(arg845_1, (112, ), (1, ))
    assert_size_stride(arg846_1, (112, ), (1, ))
    assert_size_stride(arg847_1, (), ())
    assert_size_stride(arg848_1, (112, ), (1, ))
    assert_size_stride(arg849_1, (112, ), (1, ))
    assert_size_stride(arg850_1, (), ())
    assert_size_stride(arg851_1, (112, ), (1, ))
    assert_size_stride(arg852_1, (112, ), (1, ))
    assert_size_stride(arg853_1, (), ())
    assert_size_stride(arg854_1, (112, ), (1, ))
    assert_size_stride(arg855_1, (112, ), (1, ))
    assert_size_stride(arg856_1, (), ())
    assert_size_stride(arg857_1, (112, ), (1, ))
    assert_size_stride(arg858_1, (112, ), (1, ))
    assert_size_stride(arg859_1, (), ())
    assert_size_stride(arg860_1, (112, ), (1, ))
    assert_size_stride(arg861_1, (112, ), (1, ))
    assert_size_stride(arg862_1, (), ())
    assert_size_stride(arg863_1, (112, ), (1, ))
    assert_size_stride(arg864_1, (112, ), (1, ))
    assert_size_stride(arg865_1, (), ())
    assert_size_stride(arg866_1, (2048, ), (1, ))
    assert_size_stride(arg867_1, (2048, ), (1, ))
    assert_size_stride(arg868_1, (), ())
    assert_size_stride(arg869_1, (896, ), (1, ))
    assert_size_stride(arg870_1, (896, ), (1, ))
    assert_size_stride(arg871_1, (), ())
    assert_size_stride(arg872_1, (112, ), (1, ))
    assert_size_stride(arg873_1, (112, ), (1, ))
    assert_size_stride(arg874_1, (), ())
    assert_size_stride(arg875_1, (112, ), (1, ))
    assert_size_stride(arg876_1, (112, ), (1, ))
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (112, ), (1, ))
    assert_size_stride(arg879_1, (112, ), (1, ))
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (112, ), (1, ))
    assert_size_stride(arg882_1, (112, ), (1, ))
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (112, ), (1, ))
    assert_size_stride(arg885_1, (112, ), (1, ))
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (112, ), (1, ))
    assert_size_stride(arg888_1, (112, ), (1, ))
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (112, ), (1, ))
    assert_size_stride(arg891_1, (112, ), (1, ))
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (2048, ), (1, ))
    assert_size_stride(arg894_1, (2048, ), (1, ))
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg896_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg896_1
        buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, grid=grid(192, 49), stream=stream0)
        del arg0_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg449_1, arg450_1, arg1_1, arg2_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1_1
        del arg2_1
        del arg449_1
        del arg450_1
        buf4 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 112, 56, 56), (351232, 3136, 56, 1))
        del arg3_1
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg452_1, arg453_1, arg4_1, arg5_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg452_1
        del arg453_1
        del arg4_1
        del arg5_1
        buf7 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf6, buf7, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf8 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg6_1, buf8, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf10 = buf7; del buf7  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf6, buf10, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf11 = buf8; del buf8  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg9_1, buf11, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg9_1
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf13 = buf10; del buf10  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf6, buf13, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf14 = buf11; del buf11  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg12_1, buf14, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg12_1
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf16 = buf13; del buf13  # reuse
        # Source Nodes: [sp_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf6, buf16, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf17 = buf14; del buf14  # reuse
        # Source Nodes: [sp_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg15_1, buf17, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg15_1
        # Source Nodes: [sp_13], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf19 = buf16; del buf16  # reuse
        # Source Nodes: [sp_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf6, buf19, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf20 = buf17; del buf17  # reuse
        # Source Nodes: [sp_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg18_1, buf20, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg18_1
        # Source Nodes: [sp_17], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf22 = buf19; del buf19  # reuse
        # Source Nodes: [sp_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf6, buf22, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf23 = buf20; del buf20  # reuse
        # Source Nodes: [sp_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg21_1, buf23, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg21_1
        # Source Nodes: [sp_21], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf25 = buf22; del buf22  # reuse
        # Source Nodes: [sp_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf6, buf25, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf26 = buf23; del buf23  # reuse
        # Source Nodes: [sp_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg24_1, buf26, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg24_1
        # Source Nodes: [sp_25], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf25, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 14, 56, 56), (43904, 3136, 56, 1))
        del buf25
        buf36 = empty((8, 112, 56, 56), device='cuda', dtype=torch.float32)
        buf28 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_13.run(buf6, buf28, 351232, grid=grid(351232), stream=stream0)
        buf29 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf9, arg455_1, arg456_1, arg7_1, arg8_1, buf29, 351232, grid=grid(351232), stream=stream0)
        del arg455_1
        del arg456_1
        del arg7_1
        del arg8_1
        del buf9
        buf30 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf12, arg458_1, arg459_1, arg10_1, arg11_1, buf30, 351232, grid=grid(351232), stream=stream0)
        del arg10_1
        del arg11_1
        del arg458_1
        del arg459_1
        del buf12
        buf31 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf15, arg461_1, arg462_1, arg13_1, arg14_1, buf31, 351232, grid=grid(351232), stream=stream0)
        del arg13_1
        del arg14_1
        del arg461_1
        del arg462_1
        del buf15
        buf32 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        # Source Nodes: [sp_14, sp_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf18, arg464_1, arg465_1, arg16_1, arg17_1, buf32, 351232, grid=grid(351232), stream=stream0)
        del arg16_1
        del arg17_1
        del arg464_1
        del arg465_1
        del buf18
        buf33 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        # Source Nodes: [sp_18, sp_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf21, arg467_1, arg468_1, arg19_1, arg20_1, buf33, 351232, grid=grid(351232), stream=stream0)
        del arg19_1
        del arg20_1
        del arg467_1
        del arg468_1
        del buf21
        buf34 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        # Source Nodes: [sp_22, sp_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf24, arg470_1, arg471_1, arg22_1, arg23_1, buf34, 351232, grid=grid(351232), stream=stream0)
        del arg22_1
        del arg23_1
        del arg470_1
        del arg471_1
        del buf24
        buf35 = reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Source Nodes: [sp_26, sp_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf27, arg473_1, arg474_1, arg25_1, arg26_1, buf35, 351232, grid=grid(351232), stream=stream0)
        del arg25_1
        del arg26_1
        del arg473_1
        del arg474_1
        buf37 = reinterpret_tensor(buf6, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf6  # reuse
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf36, buf37, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg27_1
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf4, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg30_1
        buf40 = buf38; del buf38  # reuse
        buf41 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf40, arg476_1, arg477_1, arg28_1, arg29_1, buf39, arg479_1, arg480_1, arg31_1, arg32_1, buf41, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        del arg28_1
        del arg29_1
        del arg31_1
        del arg32_1
        del arg476_1
        del arg477_1
        del arg479_1
        del arg480_1
        del buf39
        del buf40
        # Source Nodes: [out_8, shortcut_2], Original ATen: [aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 112, 56, 56), (351232, 3136, 56, 1))
        del arg33_1
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf43, arg482_1, arg483_1, arg34_1, arg35_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg34_1
        del arg35_1
        del arg482_1
        del arg483_1
        buf44 = reinterpret_tensor(buf27, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf27  # reuse
        # Source Nodes: [sp_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf43, buf44, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf45 = buf26; del buf26  # reuse
        # Source Nodes: [sp_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg36_1, buf45, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg36_1
        # Source Nodes: [sp_30], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf73 = reinterpret_tensor(buf37, (8, 112, 56, 56), (351232, 3136, 56, 1), 0); del buf37  # reuse
        buf47 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        buf48 = buf44; del buf44  # reuse
        # Source Nodes: [sp_31, sp_32, sp_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf46, arg485_1, arg486_1, arg37_1, arg38_1, buf43, buf47, buf48, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg37_1
        del arg38_1
        del arg485_1
        del arg486_1
        del buf46
        buf49 = buf45; del buf45  # reuse
        # Source Nodes: [sp_33, sp_34], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg39_1, buf49, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg39_1
        # Source Nodes: [sp_33, sp_34], Original ATen: [aten.add, aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf51 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [sp_35, sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf50, arg488_1, arg489_1, arg40_1, arg41_1, buf43, buf51, buf52, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg40_1
        del arg41_1
        del arg488_1
        del arg489_1
        del buf50
        buf53 = buf49; del buf49  # reuse
        # Source Nodes: [sp_37, sp_38], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg42_1, buf53, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg42_1
        # Source Nodes: [sp_37, sp_38], Original ATen: [aten.add, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf55 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [sp_39, sp_40, sp_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf54, arg491_1, arg492_1, arg43_1, arg44_1, buf43, buf55, buf56, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg43_1
        del arg44_1
        del arg491_1
        del arg492_1
        del buf54
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg45_1, buf57, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg45_1
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten.add, aten.convolution]
        buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf59 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        buf60 = buf56; del buf56  # reuse
        # Source Nodes: [sp_43, sp_44, sp_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf58, arg494_1, arg495_1, arg46_1, arg47_1, buf43, buf59, buf60, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg46_1
        del arg47_1
        del arg494_1
        del arg495_1
        del buf58
        buf61 = buf57; del buf57  # reuse
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg48_1, buf61, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg48_1
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf63 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        buf64 = buf60; del buf60  # reuse
        # Source Nodes: [sp_47, sp_48, sp_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21.run(buf62, arg497_1, arg498_1, arg49_1, arg50_1, buf43, buf63, buf64, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg497_1
        del arg498_1
        del arg49_1
        del arg50_1
        del buf62
        buf65 = buf61; del buf61  # reuse
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg51_1, buf65, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg51_1
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten.add, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf67 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        buf68 = buf64; del buf64  # reuse
        # Source Nodes: [sp_51, sp_52, sp_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf66, arg500_1, arg501_1, arg52_1, arg53_1, buf43, buf67, buf68, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg500_1
        del arg501_1
        del arg52_1
        del arg53_1
        del buf66
        buf69 = buf65; del buf65  # reuse
        # Source Nodes: [sp_53, sp_54], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg54_1, buf69, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg54_1
        # Source Nodes: [sp_53, sp_54], Original ATen: [aten.add, aten.convolution]
        buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 14, 56, 56), (43904, 3136, 56, 1))
        del buf68
        buf71 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Source Nodes: [sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf70, arg503_1, arg504_1, arg55_1, arg56_1, buf71, 351232, grid=grid(351232), stream=stream0)
        del arg503_1
        del arg504_1
        del arg55_1
        del arg56_1
        buf72 = reinterpret_tensor(buf73, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf43, buf72, 351232, grid=grid(351232), stream=stream0)
        buf74 = reinterpret_tensor(buf43, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf43  # reuse
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf73, buf74, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf47
        del buf51
        del buf55
        del buf59
        del buf63
        del buf67
        del buf71
        del buf72
        del buf73
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg57_1
        buf76 = buf41; del buf41  # reuse
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf76, buf75, arg506_1, arg507_1, arg58_1, arg59_1, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del arg506_1
        del arg507_1
        del arg58_1
        del arg59_1
        del buf75
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 112, 56, 56), (351232, 3136, 56, 1))
        del arg60_1
        buf78 = buf77; del buf77  # reuse
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf78, arg509_1, arg510_1, arg61_1, arg62_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg509_1
        del arg510_1
        del arg61_1
        del arg62_1
        buf79 = reinterpret_tensor(buf70, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf70  # reuse
        # Source Nodes: [sp_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf78, buf79, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf80 = buf69; del buf69  # reuse
        # Source Nodes: [sp_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg63_1, buf80, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg63_1
        # Source Nodes: [sp_59], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf108 = reinterpret_tensor(buf74, (8, 112, 56, 56), (351232, 3136, 56, 1), 0); del buf74  # reuse
        buf82 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        buf83 = buf79; del buf79  # reuse
        # Source Nodes: [sp_60, sp_61, sp_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf81, arg512_1, arg513_1, arg64_1, arg65_1, buf78, buf82, buf83, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg512_1
        del arg513_1
        del arg64_1
        del arg65_1
        del buf81
        buf84 = buf80; del buf80  # reuse
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg66_1, buf84, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg66_1
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten.add, aten.convolution]
        buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf86 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        buf87 = buf83; del buf83  # reuse
        # Source Nodes: [sp_64, sp_65, sp_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf85, arg515_1, arg516_1, arg67_1, arg68_1, buf78, buf86, buf87, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg515_1
        del arg516_1
        del arg67_1
        del arg68_1
        del buf85
        buf88 = buf84; del buf84  # reuse
        # Source Nodes: [sp_66, sp_67], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg69_1, buf88, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg69_1
        # Source Nodes: [sp_66, sp_67], Original ATen: [aten.add, aten.convolution]
        buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf90 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        buf91 = buf87; del buf87  # reuse
        # Source Nodes: [sp_68, sp_69, sp_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf89, arg518_1, arg519_1, arg70_1, arg71_1, buf78, buf90, buf91, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg518_1
        del arg519_1
        del arg70_1
        del arg71_1
        del buf89
        buf92 = buf88; del buf88  # reuse
        # Source Nodes: [sp_70, sp_71], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg72_1, buf92, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg72_1
        # Source Nodes: [sp_70, sp_71], Original ATen: [aten.add, aten.convolution]
        buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf94 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        buf95 = buf91; del buf91  # reuse
        # Source Nodes: [sp_72, sp_73, sp_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf93, arg521_1, arg522_1, arg73_1, arg74_1, buf78, buf94, buf95, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg521_1
        del arg522_1
        del arg73_1
        del arg74_1
        del buf93
        buf96 = buf92; del buf92  # reuse
        # Source Nodes: [sp_74, sp_75], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg75_1, buf96, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg75_1
        # Source Nodes: [sp_74, sp_75], Original ATen: [aten.add, aten.convolution]
        buf97 = extern_kernels.convolution(buf95, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf98 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [sp_76, sp_77, sp_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21.run(buf97, arg524_1, arg525_1, arg76_1, arg77_1, buf78, buf98, buf99, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg524_1
        del arg525_1
        del arg76_1
        del arg77_1
        del buf97
        buf100 = buf96; del buf96  # reuse
        # Source Nodes: [sp_78, sp_79], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg78_1, buf100, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg78_1
        # Source Nodes: [sp_78, sp_79], Original ATen: [aten.add, aten.convolution]
        buf101 = extern_kernels.convolution(buf99, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf102 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        buf103 = buf99; del buf99  # reuse
        # Source Nodes: [sp_80, sp_81, sp_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf101, arg527_1, arg528_1, arg79_1, arg80_1, buf78, buf102, buf103, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg527_1
        del arg528_1
        del arg79_1
        del arg80_1
        del buf101
        buf104 = buf100; del buf100  # reuse
        # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg81_1, buf104, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg81_1
        # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 14, 56, 56), (43904, 3136, 56, 1))
        del buf104
        buf106 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Source Nodes: [sp_84, sp_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf105, arg530_1, arg531_1, arg82_1, arg83_1, buf106, 351232, grid=grid(351232), stream=stream0)
        del arg530_1
        del arg531_1
        del arg82_1
        del arg83_1
        buf107 = reinterpret_tensor(buf108, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf78, buf107, 351232, grid=grid(351232), stream=stream0)
        buf109 = reinterpret_tensor(buf78, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf78  # reuse
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf108, buf109, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf102
        del buf106
        del buf107
        del buf108
        del buf82
        del buf86
        del buf90
        del buf94
        del buf98
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg84_1
        del buf109
        buf111 = buf76; del buf76  # reuse
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf111, buf110, arg533_1, arg534_1, arg85_1, arg86_1, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del arg533_1
        del arg534_1
        del arg85_1
        del arg86_1
        del buf110
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 224, 56, 56), (702464, 3136, 56, 1))
        del arg87_1
        buf113 = buf112; del buf112  # reuse
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf113, arg536_1, arg537_1, arg88_1, arg89_1, 5619712, grid=grid(5619712), stream=stream0)
        del arg536_1
        del arg537_1
        del arg88_1
        del arg89_1
        buf114 = empty_strided((8, 28, 56, 56), (87808, 1, 1568, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf113, buf114, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf115 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg90_1, buf115, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg90_1
        # Source Nodes: [sp_88], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf117 = buf114; del buf114  # reuse
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf113, buf117, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf118 = buf115; del buf115  # reuse
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg93_1, buf118, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg93_1
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf120 = buf117; del buf117  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf113, buf120, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf121 = buf118; del buf118  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg96_1, buf121, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg96_1
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf120, buf121, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf123 = buf120; del buf120  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf113, buf123, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf124 = buf121; del buf121  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg99_1, buf124, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg99_1
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf126 = buf123; del buf123  # reuse
        # Source Nodes: [sp_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf113, buf126, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf127 = buf124; del buf124  # reuse
        # Source Nodes: [sp_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg102_1, buf127, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg102_1
        # Source Nodes: [sp_104], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf126, buf127, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf129 = buf126; del buf126  # reuse
        # Source Nodes: [sp_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf113, buf129, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf130 = buf127; del buf127  # reuse
        # Source Nodes: [sp_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg105_1, buf130, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg105_1
        # Source Nodes: [sp_108], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf129, buf130, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf132 = buf129; del buf129  # reuse
        # Source Nodes: [sp_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf113, buf132, 224, 3136, grid=grid(224, 3136), stream=stream0)
        buf133 = buf130; del buf130  # reuse
        # Source Nodes: [sp_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg108_1, buf133, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg108_1
        # Source Nodes: [sp_112], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf143 = empty((8, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf135 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_34.run(buf113, buf135, 175616, grid=grid(175616), stream=stream0)
        del buf113
        buf136 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        # Source Nodes: [sp_89, sp_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf116, arg539_1, arg540_1, arg91_1, arg92_1, buf136, 175616, grid=grid(175616), stream=stream0)
        del arg539_1
        del arg540_1
        del arg91_1
        del arg92_1
        del buf116
        buf137 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf119, arg542_1, arg543_1, arg94_1, arg95_1, buf137, 175616, grid=grid(175616), stream=stream0)
        del arg542_1
        del arg543_1
        del arg94_1
        del arg95_1
        del buf119
        buf138 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf122, arg545_1, arg546_1, arg97_1, arg98_1, buf138, 175616, grid=grid(175616), stream=stream0)
        del arg545_1
        del arg546_1
        del arg97_1
        del arg98_1
        del buf122
        buf139 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf125, arg548_1, arg549_1, arg100_1, arg101_1, buf139, 175616, grid=grid(175616), stream=stream0)
        del arg100_1
        del arg101_1
        del arg548_1
        del arg549_1
        del buf125
        buf140 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        # Source Nodes: [sp_105, sp_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf128, arg551_1, arg552_1, arg103_1, arg104_1, buf140, 175616, grid=grid(175616), stream=stream0)
        del arg103_1
        del arg104_1
        del arg551_1
        del arg552_1
        del buf128
        buf141 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        # Source Nodes: [sp_109, sp_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf131, arg554_1, arg555_1, arg106_1, arg107_1, buf141, 175616, grid=grid(175616), stream=stream0)
        del arg106_1
        del arg107_1
        del arg554_1
        del arg555_1
        del buf131
        buf142 = reinterpret_tensor(buf143, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Source Nodes: [sp_113, sp_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf134, arg557_1, arg558_1, arg109_1, arg110_1, buf142, 175616, grid=grid(175616), stream=stream0)
        del arg109_1
        del arg110_1
        del arg557_1
        del arg558_1
        buf144 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf143, buf144, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf135
        del buf136
        del buf137
        del buf138
        del buf139
        del buf140
        del buf141
        del buf142
        del buf143
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg111_1
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf111, arg114_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg114_1
        del buf111
        buf147 = buf145; del buf145  # reuse
        buf148 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf147, arg560_1, arg561_1, arg112_1, arg113_1, buf146, arg563_1, arg564_1, arg115_1, arg116_1, buf148, 4096, 784, grid=grid(4096, 784), stream=stream0)
        del arg112_1
        del arg113_1
        del arg115_1
        del arg116_1
        del arg560_1
        del arg561_1
        del arg563_1
        del arg564_1
        del buf146
        del buf147
        # Source Nodes: [out_32, shortcut_6], Original ATen: [aten.convolution, aten.relu]
        buf149 = extern_kernels.convolution(buf148, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 224, 28, 28), (175616, 784, 28, 1))
        del arg117_1
        buf150 = buf149; del buf149  # reuse
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf150, arg566_1, arg567_1, arg118_1, arg119_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg118_1
        del arg119_1
        del arg566_1
        del arg567_1
        buf151 = reinterpret_tensor(buf134, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf134  # reuse
        # Source Nodes: [sp_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf150, buf151, 224, 784, grid=grid(224, 784), stream=stream0)
        buf152 = buf133; del buf133  # reuse
        # Source Nodes: [sp_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg120_1, buf152, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg120_1
        # Source Nodes: [sp_117], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf151, buf152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf180 = reinterpret_tensor(buf144, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf144  # reuse
        buf154 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf155 = buf151; del buf151  # reuse
        # Source Nodes: [sp_118, sp_119, sp_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf153, arg569_1, arg570_1, arg121_1, arg122_1, buf150, buf154, buf155, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg121_1
        del arg122_1
        del arg569_1
        del arg570_1
        del buf153
        buf156 = buf152; del buf152  # reuse
        # Source Nodes: [sp_120, sp_121], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg123_1, buf156, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg123_1
        # Source Nodes: [sp_120, sp_121], Original ATen: [aten.add, aten.convolution]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf158 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf159 = buf155; del buf155  # reuse
        # Source Nodes: [sp_122, sp_123, sp_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_41.run(buf157, arg572_1, arg573_1, arg124_1, arg125_1, buf150, buf158, buf159, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg124_1
        del arg125_1
        del arg572_1
        del arg573_1
        del buf157
        buf160 = buf156; del buf156  # reuse
        # Source Nodes: [sp_124, sp_125], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg126_1, buf160, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg126_1
        # Source Nodes: [sp_124, sp_125], Original ATen: [aten.add, aten.convolution]
        buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf162 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf163 = buf159; del buf159  # reuse
        # Source Nodes: [sp_126, sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42.run(buf161, arg575_1, arg576_1, arg127_1, arg128_1, buf150, buf162, buf163, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg127_1
        del arg128_1
        del arg575_1
        del arg576_1
        del buf161
        buf164 = buf160; del buf160  # reuse
        # Source Nodes: [sp_128, sp_129], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg129_1, buf164, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg129_1
        # Source Nodes: [sp_128, sp_129], Original ATen: [aten.add, aten.convolution]
        buf165 = extern_kernels.convolution(buf163, buf164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf166 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf167 = buf163; del buf163  # reuse
        # Source Nodes: [sp_130, sp_131, sp_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf165, arg578_1, arg579_1, arg130_1, arg131_1, buf150, buf166, buf167, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg130_1
        del arg131_1
        del arg578_1
        del arg579_1
        del buf165
        buf168 = buf164; del buf164  # reuse
        # Source Nodes: [sp_132, sp_133], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg132_1, buf168, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg132_1
        # Source Nodes: [sp_132, sp_133], Original ATen: [aten.add, aten.convolution]
        buf169 = extern_kernels.convolution(buf167, buf168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf170 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf171 = buf167; del buf167  # reuse
        # Source Nodes: [sp_134, sp_135, sp_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf169, arg581_1, arg582_1, arg133_1, arg134_1, buf150, buf170, buf171, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg133_1
        del arg134_1
        del arg581_1
        del arg582_1
        del buf169
        buf172 = buf168; del buf168  # reuse
        # Source Nodes: [sp_136, sp_137], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg135_1, buf172, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg135_1
        # Source Nodes: [sp_136, sp_137], Original ATen: [aten.add, aten.convolution]
        buf173 = extern_kernels.convolution(buf171, buf172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf174 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf175 = buf171; del buf171  # reuse
        # Source Nodes: [sp_138, sp_139, sp_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf173, arg584_1, arg585_1, arg136_1, arg137_1, buf150, buf174, buf175, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg136_1
        del arg137_1
        del arg584_1
        del arg585_1
        del buf173
        buf176 = buf172; del buf172  # reuse
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg138_1, buf176, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg138_1
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten.add, aten.convolution]
        buf177 = extern_kernels.convolution(buf175, buf176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf175
        buf178 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Source Nodes: [sp_142, sp_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf177, arg587_1, arg588_1, arg139_1, arg140_1, buf178, 175616, grid=grid(175616), stream=stream0)
        del arg139_1
        del arg140_1
        del arg587_1
        del arg588_1
        buf179 = reinterpret_tensor(buf180, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf150, buf179, 175616, grid=grid(175616), stream=stream0)
        buf181 = reinterpret_tensor(buf150, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf150  # reuse
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf180, buf181, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf154
        del buf158
        del buf162
        del buf166
        del buf170
        del buf174
        del buf178
        del buf179
        del buf180
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg141_1
        buf183 = buf148; del buf148  # reuse
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf183, buf182, arg590_1, arg591_1, arg142_1, arg143_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg142_1
        del arg143_1
        del arg590_1
        del arg591_1
        del buf182
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 224, 28, 28), (175616, 784, 28, 1))
        del arg144_1
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf185, arg593_1, arg594_1, arg145_1, arg146_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg145_1
        del arg146_1
        del arg593_1
        del arg594_1
        buf186 = reinterpret_tensor(buf177, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf177  # reuse
        # Source Nodes: [sp_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf185, buf186, 224, 784, grid=grid(224, 784), stream=stream0)
        buf187 = buf176; del buf176  # reuse
        # Source Nodes: [sp_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg147_1, buf187, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg147_1
        # Source Nodes: [sp_146], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf186, buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf215 = reinterpret_tensor(buf181, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf181  # reuse
        buf189 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf190 = buf186; del buf186  # reuse
        # Source Nodes: [sp_147, sp_148, sp_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf188, arg596_1, arg597_1, arg148_1, arg149_1, buf185, buf189, buf190, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg148_1
        del arg149_1
        del arg596_1
        del arg597_1
        del buf188
        buf191 = buf187; del buf187  # reuse
        # Source Nodes: [sp_149, sp_150], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg150_1, buf191, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg150_1
        # Source Nodes: [sp_149, sp_150], Original ATen: [aten.add, aten.convolution]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf193 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf194 = buf190; del buf190  # reuse
        # Source Nodes: [sp_151, sp_152, sp_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_41.run(buf192, arg599_1, arg600_1, arg151_1, arg152_1, buf185, buf193, buf194, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg151_1
        del arg152_1
        del arg599_1
        del arg600_1
        del buf192
        buf195 = buf191; del buf191  # reuse
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg153_1, buf195, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg153_1
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten.add, aten.convolution]
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf197 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf198 = buf194; del buf194  # reuse
        # Source Nodes: [sp_155, sp_156, sp_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42.run(buf196, arg602_1, arg603_1, arg154_1, arg155_1, buf185, buf197, buf198, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg154_1
        del arg155_1
        del arg602_1
        del arg603_1
        del buf196
        buf199 = buf195; del buf195  # reuse
        # Source Nodes: [sp_157, sp_158], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg156_1, buf199, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg156_1
        # Source Nodes: [sp_157, sp_158], Original ATen: [aten.add, aten.convolution]
        buf200 = extern_kernels.convolution(buf198, buf199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf201 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf202 = buf198; del buf198  # reuse
        # Source Nodes: [sp_159, sp_160, sp_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf200, arg605_1, arg606_1, arg157_1, arg158_1, buf185, buf201, buf202, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg157_1
        del arg158_1
        del arg605_1
        del arg606_1
        del buf200
        buf203 = buf199; del buf199  # reuse
        # Source Nodes: [sp_161, sp_162], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg159_1, buf203, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg159_1
        # Source Nodes: [sp_161, sp_162], Original ATen: [aten.add, aten.convolution]
        buf204 = extern_kernels.convolution(buf202, buf203, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf205 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf206 = buf202; del buf202  # reuse
        # Source Nodes: [sp_163, sp_164, sp_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf204, arg608_1, arg609_1, arg160_1, arg161_1, buf185, buf205, buf206, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg160_1
        del arg161_1
        del arg608_1
        del arg609_1
        del buf204
        buf207 = buf203; del buf203  # reuse
        # Source Nodes: [sp_165, sp_166], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg162_1, buf207, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg162_1
        # Source Nodes: [sp_165, sp_166], Original ATen: [aten.add, aten.convolution]
        buf208 = extern_kernels.convolution(buf206, buf207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf209 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf210 = buf206; del buf206  # reuse
        # Source Nodes: [sp_167, sp_168, sp_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf208, arg611_1, arg612_1, arg163_1, arg164_1, buf185, buf209, buf210, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg163_1
        del arg164_1
        del arg611_1
        del arg612_1
        del buf208
        buf211 = buf207; del buf207  # reuse
        # Source Nodes: [sp_169, sp_170], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg165_1, buf211, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg165_1
        # Source Nodes: [sp_169, sp_170], Original ATen: [aten.add, aten.convolution]
        buf212 = extern_kernels.convolution(buf210, buf211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf210
        buf213 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf212, arg614_1, arg615_1, arg166_1, arg167_1, buf213, 175616, grid=grid(175616), stream=stream0)
        del arg166_1
        del arg167_1
        del arg614_1
        del arg615_1
        buf214 = reinterpret_tensor(buf215, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf185, buf214, 175616, grid=grid(175616), stream=stream0)
        buf216 = reinterpret_tensor(buf185, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf185  # reuse
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf215, buf216, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf189
        del buf193
        del buf197
        del buf201
        del buf205
        del buf209
        del buf213
        del buf214
        del buf215
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg168_1
        buf218 = buf183; del buf183  # reuse
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf218, buf217, arg617_1, arg618_1, arg169_1, arg170_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg169_1
        del arg170_1
        del arg617_1
        del arg618_1
        del buf217
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 224, 28, 28), (175616, 784, 28, 1))
        del arg171_1
        buf220 = buf219; del buf219  # reuse
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf220, arg620_1, arg621_1, arg172_1, arg173_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg172_1
        del arg173_1
        del arg620_1
        del arg621_1
        buf221 = reinterpret_tensor(buf212, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf212  # reuse
        # Source Nodes: [sp_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf220, buf221, 224, 784, grid=grid(224, 784), stream=stream0)
        buf222 = buf211; del buf211  # reuse
        # Source Nodes: [sp_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg174_1, buf222, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg174_1
        # Source Nodes: [sp_175], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf221, buf222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf250 = reinterpret_tensor(buf216, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf216  # reuse
        buf224 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf225 = buf221; del buf221  # reuse
        # Source Nodes: [sp_176, sp_177, sp_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf223, arg623_1, arg624_1, arg175_1, arg176_1, buf220, buf224, buf225, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg175_1
        del arg176_1
        del arg623_1
        del arg624_1
        del buf223
        buf226 = buf222; del buf222  # reuse
        # Source Nodes: [sp_178, sp_179], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg177_1, buf226, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg177_1
        # Source Nodes: [sp_178, sp_179], Original ATen: [aten.add, aten.convolution]
        buf227 = extern_kernels.convolution(buf225, buf226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf228 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf229 = buf225; del buf225  # reuse
        # Source Nodes: [sp_180, sp_181, sp_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_41.run(buf227, arg626_1, arg627_1, arg178_1, arg179_1, buf220, buf228, buf229, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg178_1
        del arg179_1
        del arg626_1
        del arg627_1
        del buf227
        buf230 = buf226; del buf226  # reuse
        # Source Nodes: [sp_182, sp_183], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg180_1, buf230, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg180_1
        # Source Nodes: [sp_182, sp_183], Original ATen: [aten.add, aten.convolution]
        buf231 = extern_kernels.convolution(buf229, buf230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf232 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf233 = buf229; del buf229  # reuse
        # Source Nodes: [sp_184, sp_185, sp_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42.run(buf231, arg629_1, arg630_1, arg181_1, arg182_1, buf220, buf232, buf233, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg181_1
        del arg182_1
        del arg629_1
        del arg630_1
        del buf231
        buf234 = buf230; del buf230  # reuse
        # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg183_1, buf234, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg183_1
        # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf236 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf237 = buf233; del buf233  # reuse
        # Source Nodes: [sp_188, sp_189, sp_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf235, arg632_1, arg633_1, arg184_1, arg185_1, buf220, buf236, buf237, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg184_1
        del arg185_1
        del arg632_1
        del arg633_1
        del buf235
        buf238 = buf234; del buf234  # reuse
        # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg186_1, buf238, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg186_1
        # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
        buf239 = extern_kernels.convolution(buf237, buf238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf240 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf241 = buf237; del buf237  # reuse
        # Source Nodes: [sp_192, sp_193, sp_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf239, arg635_1, arg636_1, arg187_1, arg188_1, buf220, buf240, buf241, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg187_1
        del arg188_1
        del arg635_1
        del arg636_1
        del buf239
        buf242 = buf238; del buf238  # reuse
        # Source Nodes: [sp_194, sp_195], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg189_1, buf242, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg189_1
        # Source Nodes: [sp_194, sp_195], Original ATen: [aten.add, aten.convolution]
        buf243 = extern_kernels.convolution(buf241, buf242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf244 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf245 = buf241; del buf241  # reuse
        # Source Nodes: [sp_196, sp_197, sp_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf243, arg638_1, arg639_1, arg190_1, arg191_1, buf220, buf244, buf245, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg190_1
        del arg191_1
        del arg638_1
        del arg639_1
        del buf243
        buf246 = buf242; del buf242  # reuse
        # Source Nodes: [sp_198, sp_199], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg192_1, buf246, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg192_1
        # Source Nodes: [sp_198, sp_199], Original ATen: [aten.add, aten.convolution]
        buf247 = extern_kernels.convolution(buf245, buf246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf245
        del buf246
        buf248 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Source Nodes: [sp_200, sp_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf247, arg641_1, arg642_1, arg193_1, arg194_1, buf248, 175616, grid=grid(175616), stream=stream0)
        del arg193_1
        del arg194_1
        del arg641_1
        del arg642_1
        buf249 = reinterpret_tensor(buf250, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf220, buf249, 175616, grid=grid(175616), stream=stream0)
        buf251 = reinterpret_tensor(buf220, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf220  # reuse
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf250, buf251, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf224
        del buf228
        del buf232
        del buf236
        del buf240
        del buf244
        del buf248
        del buf249
        del buf250
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg195_1
        del buf251
        buf253 = buf218; del buf218  # reuse
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf253, buf252, arg644_1, arg645_1, arg196_1, arg197_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg196_1
        del arg197_1
        del arg644_1
        del arg645_1
        del buf252
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 448, 28, 28), (351232, 784, 28, 1))
        del arg198_1
        buf255 = buf254; del buf254  # reuse
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf255, arg647_1, arg648_1, arg199_1, arg200_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg199_1
        del arg200_1
        del arg647_1
        del arg648_1
        buf256 = reinterpret_tensor(buf105, (8, 56, 28, 28), (43904, 1, 1568, 56), 0); del buf105  # reuse
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf255, buf256, 448, 784, grid=grid(448, 784), stream=stream0)
        buf257 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg201_1, buf257, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg201_1
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf256, buf257, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf259 = buf256; del buf256  # reuse
        # Source Nodes: [sp_208], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf255, buf259, 448, 784, grid=grid(448, 784), stream=stream0)
        buf260 = buf257; del buf257  # reuse
        # Source Nodes: [sp_208], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg204_1, buf260, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg204_1
        # Source Nodes: [sp_208], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf259, buf260, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf262 = buf259; del buf259  # reuse
        # Source Nodes: [sp_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf255, buf262, 448, 784, grid=grid(448, 784), stream=stream0)
        buf263 = buf260; del buf260  # reuse
        # Source Nodes: [sp_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg207_1, buf263, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg207_1
        # Source Nodes: [sp_212], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf262, buf263, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf265 = buf262; del buf262  # reuse
        # Source Nodes: [sp_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf255, buf265, 448, 784, grid=grid(448, 784), stream=stream0)
        buf266 = buf263; del buf263  # reuse
        # Source Nodes: [sp_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg210_1, buf266, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg210_1
        # Source Nodes: [sp_216], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf265, buf266, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf268 = buf265; del buf265  # reuse
        # Source Nodes: [sp_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf255, buf268, 448, 784, grid=grid(448, 784), stream=stream0)
        buf269 = buf266; del buf266  # reuse
        # Source Nodes: [sp_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg213_1, buf269, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg213_1
        # Source Nodes: [sp_220], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf268, buf269, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf271 = buf268; del buf268  # reuse
        # Source Nodes: [sp_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf255, buf271, 448, 784, grid=grid(448, 784), stream=stream0)
        buf272 = buf269; del buf269  # reuse
        # Source Nodes: [sp_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg216_1, buf272, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg216_1
        # Source Nodes: [sp_224], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf271, buf272, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf274 = buf271; del buf271  # reuse
        # Source Nodes: [sp_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf255, buf274, 448, 784, grid=grid(448, 784), stream=stream0)
        buf275 = buf272; del buf272  # reuse
        # Source Nodes: [sp_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg219_1, buf275, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg219_1
        # Source Nodes: [sp_228], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf274, buf275, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf285 = reinterpret_tensor(buf132, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf132  # reuse
        buf277 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_57.run(buf255, buf277, 87808, grid=grid(87808), stream=stream0)
        del buf255
        buf278 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf258, arg650_1, arg651_1, arg202_1, arg203_1, buf278, 87808, grid=grid(87808), stream=stream0)
        del arg202_1
        del arg203_1
        del arg650_1
        del arg651_1
        del buf258
        buf279 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Source Nodes: [sp_209, sp_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf261, arg653_1, arg654_1, arg205_1, arg206_1, buf279, 87808, grid=grid(87808), stream=stream0)
        del arg205_1
        del arg206_1
        del arg653_1
        del arg654_1
        del buf261
        buf280 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Source Nodes: [sp_213, sp_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf264, arg656_1, arg657_1, arg208_1, arg209_1, buf280, 87808, grid=grid(87808), stream=stream0)
        del arg208_1
        del arg209_1
        del arg656_1
        del arg657_1
        del buf264
        buf281 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Source Nodes: [sp_217, sp_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf267, arg659_1, arg660_1, arg211_1, arg212_1, buf281, 87808, grid=grid(87808), stream=stream0)
        del arg211_1
        del arg212_1
        del arg659_1
        del arg660_1
        del buf267
        buf282 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Source Nodes: [sp_221, sp_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf270, arg662_1, arg663_1, arg214_1, arg215_1, buf282, 87808, grid=grid(87808), stream=stream0)
        del arg214_1
        del arg215_1
        del arg662_1
        del arg663_1
        del buf270
        buf283 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Source Nodes: [sp_225, sp_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf273, arg665_1, arg666_1, arg217_1, arg218_1, buf283, 87808, grid=grid(87808), stream=stream0)
        del arg217_1
        del arg218_1
        del arg665_1
        del arg666_1
        del buf273
        buf284 = reinterpret_tensor(buf285, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_229, sp_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf276, arg668_1, arg669_1, arg220_1, arg221_1, buf284, 87808, grid=grid(87808), stream=stream0)
        del arg220_1
        del arg221_1
        del arg668_1
        del arg669_1
        buf286 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf285, buf286, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf277
        del buf278
        del buf279
        del buf280
        del buf281
        del buf282
        del buf283
        del buf284
        del buf285
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg222_1
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf253, arg225_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg225_1
        del buf253
        buf289 = buf287; del buf287  # reuse
        buf290 = reinterpret_tensor(buf4, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf4  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_60.run(buf289, arg671_1, arg672_1, arg223_1, arg224_1, buf288, arg674_1, arg675_1, arg226_1, arg227_1, buf290, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del arg223_1
        del arg224_1
        del arg226_1
        del arg227_1
        del arg671_1
        del arg672_1
        del arg674_1
        del arg675_1
        del buf288
        del buf289
        # Source Nodes: [out_64, shortcut_11], Original ATen: [aten.convolution, aten.relu]
        buf291 = extern_kernels.convolution(buf290, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 448, 14, 14), (87808, 196, 14, 1))
        del arg228_1
        buf292 = buf291; del buf291  # reuse
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf292, arg677_1, arg678_1, arg229_1, arg230_1, 702464, grid=grid(702464), stream=stream0)
        del arg229_1
        del arg230_1
        del arg677_1
        del arg678_1
        buf293 = reinterpret_tensor(buf276, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf276  # reuse
        # Source Nodes: [sp_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf292, buf293, 448, 196, grid=grid(448, 196), stream=stream0)
        buf294 = buf275; del buf275  # reuse
        # Source Nodes: [sp_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg231_1, buf294, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg231_1
        # Source Nodes: [sp_233], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf293, buf294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf322 = reinterpret_tensor(buf286, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf286  # reuse
        buf296 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf297 = buf293; del buf293  # reuse
        # Source Nodes: [sp_234, sp_235, sp_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf295, arg680_1, arg681_1, arg232_1, arg233_1, buf292, buf296, buf297, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg232_1
        del arg233_1
        del arg680_1
        del arg681_1
        del buf295
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [sp_236, sp_237], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg234_1, buf298, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg234_1
        # Source Nodes: [sp_236, sp_237], Original ATen: [aten.add, aten.convolution]
        buf299 = extern_kernels.convolution(buf297, buf298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf300 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf301 = buf297; del buf297  # reuse
        # Source Nodes: [sp_238, sp_239, sp_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64.run(buf299, arg683_1, arg684_1, arg235_1, arg236_1, buf292, buf300, buf301, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg235_1
        del arg236_1
        del arg683_1
        del arg684_1
        del buf299
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [sp_240, sp_241], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg237_1, buf302, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg237_1
        # Source Nodes: [sp_240, sp_241], Original ATen: [aten.add, aten.convolution]
        buf303 = extern_kernels.convolution(buf301, buf302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf304 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf305 = buf301; del buf301  # reuse
        # Source Nodes: [sp_242, sp_243, sp_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf303, arg686_1, arg687_1, arg238_1, arg239_1, buf292, buf304, buf305, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg238_1
        del arg239_1
        del arg686_1
        del arg687_1
        del buf303
        buf306 = buf302; del buf302  # reuse
        # Source Nodes: [sp_244, sp_245], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg240_1, buf306, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg240_1
        # Source Nodes: [sp_244, sp_245], Original ATen: [aten.add, aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf308 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf309 = buf305; del buf305  # reuse
        # Source Nodes: [sp_246, sp_247, sp_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66.run(buf307, arg689_1, arg690_1, arg241_1, arg242_1, buf292, buf308, buf309, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg241_1
        del arg242_1
        del arg689_1
        del arg690_1
        del buf307
        buf310 = buf306; del buf306  # reuse
        # Source Nodes: [sp_248, sp_249], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg243_1, buf310, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg243_1
        # Source Nodes: [sp_248, sp_249], Original ATen: [aten.add, aten.convolution]
        buf311 = extern_kernels.convolution(buf309, buf310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf312 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf313 = buf309; del buf309  # reuse
        # Source Nodes: [sp_250, sp_251, sp_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67.run(buf311, arg692_1, arg693_1, arg244_1, arg245_1, buf292, buf312, buf313, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg244_1
        del arg245_1
        del arg692_1
        del arg693_1
        del buf311
        buf314 = buf310; del buf310  # reuse
        # Source Nodes: [sp_252, sp_253], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg246_1, buf314, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg246_1
        # Source Nodes: [sp_252, sp_253], Original ATen: [aten.add, aten.convolution]
        buf315 = extern_kernels.convolution(buf313, buf314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf316 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf317 = buf313; del buf313  # reuse
        # Source Nodes: [sp_254, sp_255, sp_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68.run(buf315, arg695_1, arg696_1, arg247_1, arg248_1, buf292, buf316, buf317, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg247_1
        del arg248_1
        del arg695_1
        del arg696_1
        del buf315
        buf318 = buf314; del buf314  # reuse
        # Source Nodes: [sp_256, sp_257], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg249_1, buf318, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg249_1
        # Source Nodes: [sp_256, sp_257], Original ATen: [aten.add, aten.convolution]
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 56, 14, 14), (10976, 196, 14, 1))
        del buf317
        buf320 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_258, sp_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf319, arg698_1, arg699_1, arg250_1, arg251_1, buf320, 87808, grid=grid(87808), stream=stream0)
        del arg250_1
        del arg251_1
        del arg698_1
        del arg699_1
        buf321 = reinterpret_tensor(buf322, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf292, buf321, 87808, grid=grid(87808), stream=stream0)
        buf323 = reinterpret_tensor(buf292, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf292  # reuse
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf322, buf323, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf296
        del buf300
        del buf304
        del buf308
        del buf312
        del buf316
        del buf320
        del buf321
        del buf322
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg252_1
        buf325 = buf290; del buf290  # reuse
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70.run(buf325, buf324, arg701_1, arg702_1, arg253_1, arg254_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg253_1
        del arg254_1
        del arg701_1
        del arg702_1
        del buf324
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 448, 14, 14), (87808, 196, 14, 1))
        del arg255_1
        buf327 = buf326; del buf326  # reuse
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf327, arg704_1, arg705_1, arg256_1, arg257_1, 702464, grid=grid(702464), stream=stream0)
        del arg256_1
        del arg257_1
        del arg704_1
        del arg705_1
        buf328 = reinterpret_tensor(buf319, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf319  # reuse
        # Source Nodes: [sp_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf327, buf328, 448, 196, grid=grid(448, 196), stream=stream0)
        buf329 = buf318; del buf318  # reuse
        # Source Nodes: [sp_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg258_1, buf329, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg258_1
        # Source Nodes: [sp_262], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf328, buf329, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf357 = reinterpret_tensor(buf323, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf323  # reuse
        buf331 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf332 = buf328; del buf328  # reuse
        # Source Nodes: [sp_263, sp_264, sp_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf330, arg707_1, arg708_1, arg259_1, arg260_1, buf327, buf331, buf332, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg259_1
        del arg260_1
        del arg707_1
        del arg708_1
        del buf330
        buf333 = buf329; del buf329  # reuse
        # Source Nodes: [sp_265, sp_266], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg261_1, buf333, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg261_1
        # Source Nodes: [sp_265, sp_266], Original ATen: [aten.add, aten.convolution]
        buf334 = extern_kernels.convolution(buf332, buf333, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf335 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf336 = buf332; del buf332  # reuse
        # Source Nodes: [sp_267, sp_268, sp_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64.run(buf334, arg710_1, arg711_1, arg262_1, arg263_1, buf327, buf335, buf336, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg262_1
        del arg263_1
        del arg710_1
        del arg711_1
        del buf334
        buf337 = buf333; del buf333  # reuse
        # Source Nodes: [sp_269, sp_270], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg264_1, buf337, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg264_1
        # Source Nodes: [sp_269, sp_270], Original ATen: [aten.add, aten.convolution]
        buf338 = extern_kernels.convolution(buf336, buf337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf339 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf340 = buf336; del buf336  # reuse
        # Source Nodes: [sp_271, sp_272, sp_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf338, arg713_1, arg714_1, arg265_1, arg266_1, buf327, buf339, buf340, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg265_1
        del arg266_1
        del arg713_1
        del arg714_1
        del buf338
        buf341 = buf337; del buf337  # reuse
        # Source Nodes: [sp_273, sp_274], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg267_1, buf341, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg267_1
        # Source Nodes: [sp_273, sp_274], Original ATen: [aten.add, aten.convolution]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf343 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf344 = buf340; del buf340  # reuse
        # Source Nodes: [sp_275, sp_276, sp_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66.run(buf342, arg716_1, arg717_1, arg268_1, arg269_1, buf327, buf343, buf344, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg268_1
        del arg269_1
        del arg716_1
        del arg717_1
        del buf342
        buf345 = buf341; del buf341  # reuse
        # Source Nodes: [sp_277, sp_278], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg270_1, buf345, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg270_1
        # Source Nodes: [sp_277, sp_278], Original ATen: [aten.add, aten.convolution]
        buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf347 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf348 = buf344; del buf344  # reuse
        # Source Nodes: [sp_279, sp_280, sp_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67.run(buf346, arg719_1, arg720_1, arg271_1, arg272_1, buf327, buf347, buf348, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg271_1
        del arg272_1
        del arg719_1
        del arg720_1
        del buf346
        buf349 = buf345; del buf345  # reuse
        # Source Nodes: [sp_281, sp_282], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg273_1, buf349, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg273_1
        # Source Nodes: [sp_281, sp_282], Original ATen: [aten.add, aten.convolution]
        buf350 = extern_kernels.convolution(buf348, buf349, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf351 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf352 = buf348; del buf348  # reuse
        # Source Nodes: [sp_283, sp_284, sp_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68.run(buf350, arg722_1, arg723_1, arg274_1, arg275_1, buf327, buf351, buf352, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg274_1
        del arg275_1
        del arg722_1
        del arg723_1
        del buf350
        buf353 = buf349; del buf349  # reuse
        # Source Nodes: [sp_285, sp_286], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg276_1, buf353, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg276_1
        # Source Nodes: [sp_285, sp_286], Original ATen: [aten.add, aten.convolution]
        buf354 = extern_kernels.convolution(buf352, buf353, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 56, 14, 14), (10976, 196, 14, 1))
        del buf352
        buf355 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_287, sp_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf354, arg725_1, arg726_1, arg277_1, arg278_1, buf355, 87808, grid=grid(87808), stream=stream0)
        del arg277_1
        del arg278_1
        del arg725_1
        del arg726_1
        buf356 = reinterpret_tensor(buf357, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf327, buf356, 87808, grid=grid(87808), stream=stream0)
        buf358 = reinterpret_tensor(buf327, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf327  # reuse
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf357, buf358, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf331
        del buf335
        del buf339
        del buf343
        del buf347
        del buf351
        del buf355
        del buf356
        del buf357
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg279_1
        buf360 = buf325; del buf325  # reuse
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70.run(buf360, buf359, arg728_1, arg729_1, arg280_1, arg281_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg280_1
        del arg281_1
        del arg728_1
        del arg729_1
        del buf359
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 448, 14, 14), (87808, 196, 14, 1))
        del arg282_1
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf362, arg731_1, arg732_1, arg283_1, arg284_1, 702464, grid=grid(702464), stream=stream0)
        del arg283_1
        del arg284_1
        del arg731_1
        del arg732_1
        buf363 = reinterpret_tensor(buf354, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf354  # reuse
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf362, buf363, 448, 196, grid=grid(448, 196), stream=stream0)
        buf364 = buf353; del buf353  # reuse
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg285_1, buf364, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg285_1
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf363, buf364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf392 = reinterpret_tensor(buf358, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf358  # reuse
        buf366 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf367 = buf363; del buf363  # reuse
        # Source Nodes: [sp_292, sp_293, sp_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf365, arg734_1, arg735_1, arg286_1, arg287_1, buf362, buf366, buf367, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg286_1
        del arg287_1
        del arg734_1
        del arg735_1
        del buf365
        buf368 = buf364; del buf364  # reuse
        # Source Nodes: [sp_294, sp_295], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg288_1, buf368, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg288_1
        # Source Nodes: [sp_294, sp_295], Original ATen: [aten.add, aten.convolution]
        buf369 = extern_kernels.convolution(buf367, buf368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf370 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf371 = buf367; del buf367  # reuse
        # Source Nodes: [sp_296, sp_297, sp_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64.run(buf369, arg737_1, arg738_1, arg289_1, arg290_1, buf362, buf370, buf371, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg289_1
        del arg290_1
        del arg737_1
        del arg738_1
        del buf369
        buf372 = buf368; del buf368  # reuse
        # Source Nodes: [sp_298, sp_299], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg291_1, buf372, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg291_1
        # Source Nodes: [sp_298, sp_299], Original ATen: [aten.add, aten.convolution]
        buf373 = extern_kernels.convolution(buf371, buf372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf374 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf375 = buf371; del buf371  # reuse
        # Source Nodes: [sp_300, sp_301, sp_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf373, arg740_1, arg741_1, arg292_1, arg293_1, buf362, buf374, buf375, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg292_1
        del arg293_1
        del arg740_1
        del arg741_1
        del buf373
        buf376 = buf372; del buf372  # reuse
        # Source Nodes: [sp_302, sp_303], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg294_1, buf376, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg294_1
        # Source Nodes: [sp_302, sp_303], Original ATen: [aten.add, aten.convolution]
        buf377 = extern_kernels.convolution(buf375, buf376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf378 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf379 = buf375; del buf375  # reuse
        # Source Nodes: [sp_304, sp_305, sp_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66.run(buf377, arg743_1, arg744_1, arg295_1, arg296_1, buf362, buf378, buf379, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg295_1
        del arg296_1
        del arg743_1
        del arg744_1
        del buf377
        buf380 = buf376; del buf376  # reuse
        # Source Nodes: [sp_306, sp_307], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg297_1, buf380, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg297_1
        # Source Nodes: [sp_306, sp_307], Original ATen: [aten.add, aten.convolution]
        buf381 = extern_kernels.convolution(buf379, buf380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf382 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf383 = buf379; del buf379  # reuse
        # Source Nodes: [sp_308, sp_309, sp_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67.run(buf381, arg746_1, arg747_1, arg298_1, arg299_1, buf362, buf382, buf383, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg298_1
        del arg299_1
        del arg746_1
        del arg747_1
        del buf381
        buf384 = buf380; del buf380  # reuse
        # Source Nodes: [sp_310, sp_311], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg300_1, buf384, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg300_1
        # Source Nodes: [sp_310, sp_311], Original ATen: [aten.add, aten.convolution]
        buf385 = extern_kernels.convolution(buf383, buf384, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf386 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf387 = buf383; del buf383  # reuse
        # Source Nodes: [sp_312, sp_313, sp_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68.run(buf385, arg749_1, arg750_1, arg301_1, arg302_1, buf362, buf386, buf387, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg301_1
        del arg302_1
        del arg749_1
        del arg750_1
        del buf385
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [sp_314, sp_315], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg303_1, buf388, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg303_1
        # Source Nodes: [sp_314, sp_315], Original ATen: [aten.add, aten.convolution]
        buf389 = extern_kernels.convolution(buf387, buf388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 56, 14, 14), (10976, 196, 14, 1))
        del buf387
        buf390 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_316, sp_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf389, arg752_1, arg753_1, arg304_1, arg305_1, buf390, 87808, grid=grid(87808), stream=stream0)
        del arg304_1
        del arg305_1
        del arg752_1
        del arg753_1
        buf391 = reinterpret_tensor(buf392, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf362, buf391, 87808, grid=grid(87808), stream=stream0)
        buf393 = reinterpret_tensor(buf362, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf362  # reuse
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf392, buf393, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf366
        del buf370
        del buf374
        del buf378
        del buf382
        del buf386
        del buf390
        del buf391
        del buf392
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg306_1
        buf395 = buf360; del buf360  # reuse
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70.run(buf395, buf394, arg755_1, arg756_1, arg307_1, arg308_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg307_1
        del arg308_1
        del arg755_1
        del arg756_1
        del buf394
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 448, 14, 14), (87808, 196, 14, 1))
        del arg309_1
        buf397 = buf396; del buf396  # reuse
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf397, arg758_1, arg759_1, arg310_1, arg311_1, 702464, grid=grid(702464), stream=stream0)
        del arg310_1
        del arg311_1
        del arg758_1
        del arg759_1
        buf398 = reinterpret_tensor(buf389, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf389  # reuse
        # Source Nodes: [sp_320], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf397, buf398, 448, 196, grid=grid(448, 196), stream=stream0)
        buf399 = buf388; del buf388  # reuse
        # Source Nodes: [sp_320], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg312_1, buf399, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg312_1
        # Source Nodes: [sp_320], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf398, buf399, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf427 = reinterpret_tensor(buf393, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf393  # reuse
        buf401 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf402 = buf398; del buf398  # reuse
        # Source Nodes: [sp_321, sp_322, sp_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf400, arg761_1, arg762_1, arg313_1, arg314_1, buf397, buf401, buf402, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg313_1
        del arg314_1
        del arg761_1
        del arg762_1
        del buf400
        buf403 = buf399; del buf399  # reuse
        # Source Nodes: [sp_323, sp_324], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg315_1, buf403, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg315_1
        # Source Nodes: [sp_323, sp_324], Original ATen: [aten.add, aten.convolution]
        buf404 = extern_kernels.convolution(buf402, buf403, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf405 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf406 = buf402; del buf402  # reuse
        # Source Nodes: [sp_325, sp_326, sp_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64.run(buf404, arg764_1, arg765_1, arg316_1, arg317_1, buf397, buf405, buf406, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg316_1
        del arg317_1
        del arg764_1
        del arg765_1
        del buf404
        buf407 = buf403; del buf403  # reuse
        # Source Nodes: [sp_327, sp_328], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg318_1, buf407, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg318_1
        # Source Nodes: [sp_327, sp_328], Original ATen: [aten.add, aten.convolution]
        buf408 = extern_kernels.convolution(buf406, buf407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf409 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf410 = buf406; del buf406  # reuse
        # Source Nodes: [sp_329, sp_330, sp_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf408, arg767_1, arg768_1, arg319_1, arg320_1, buf397, buf409, buf410, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg319_1
        del arg320_1
        del arg767_1
        del arg768_1
        del buf408
        buf411 = buf407; del buf407  # reuse
        # Source Nodes: [sp_331, sp_332], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg321_1, buf411, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg321_1
        # Source Nodes: [sp_331, sp_332], Original ATen: [aten.add, aten.convolution]
        buf412 = extern_kernels.convolution(buf410, buf411, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf413 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf414 = buf410; del buf410  # reuse
        # Source Nodes: [sp_333, sp_334, sp_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66.run(buf412, arg770_1, arg771_1, arg322_1, arg323_1, buf397, buf413, buf414, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg322_1
        del arg323_1
        del arg770_1
        del arg771_1
        del buf412
        buf415 = buf411; del buf411  # reuse
        # Source Nodes: [sp_335, sp_336], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg324_1, buf415, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg324_1
        # Source Nodes: [sp_335, sp_336], Original ATen: [aten.add, aten.convolution]
        buf416 = extern_kernels.convolution(buf414, buf415, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf417 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf418 = buf414; del buf414  # reuse
        # Source Nodes: [sp_337, sp_338, sp_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67.run(buf416, arg773_1, arg774_1, arg325_1, arg326_1, buf397, buf417, buf418, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg325_1
        del arg326_1
        del arg773_1
        del arg774_1
        del buf416
        buf419 = buf415; del buf415  # reuse
        # Source Nodes: [sp_339, sp_340], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg327_1, buf419, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg327_1
        # Source Nodes: [sp_339, sp_340], Original ATen: [aten.add, aten.convolution]
        buf420 = extern_kernels.convolution(buf418, buf419, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf421 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf422 = buf418; del buf418  # reuse
        # Source Nodes: [sp_341, sp_342, sp_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68.run(buf420, arg776_1, arg777_1, arg328_1, arg329_1, buf397, buf421, buf422, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg328_1
        del arg329_1
        del arg776_1
        del arg777_1
        del buf420
        buf423 = buf419; del buf419  # reuse
        # Source Nodes: [sp_343, sp_344], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg330_1, buf423, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg330_1
        # Source Nodes: [sp_343, sp_344], Original ATen: [aten.add, aten.convolution]
        buf424 = extern_kernels.convolution(buf422, buf423, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 56, 14, 14), (10976, 196, 14, 1))
        del buf422
        buf425 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_345, sp_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf424, arg779_1, arg780_1, arg331_1, arg332_1, buf425, 87808, grid=grid(87808), stream=stream0)
        del arg331_1
        del arg332_1
        del arg779_1
        del arg780_1
        buf426 = reinterpret_tensor(buf427, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf397, buf426, 87808, grid=grid(87808), stream=stream0)
        buf428 = reinterpret_tensor(buf397, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf397  # reuse
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf427, buf428, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf401
        del buf405
        del buf409
        del buf413
        del buf417
        del buf421
        del buf425
        del buf426
        del buf427
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg333_1
        buf430 = buf395; del buf395  # reuse
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70.run(buf430, buf429, arg782_1, arg783_1, arg334_1, arg335_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg334_1
        del arg335_1
        del arg782_1
        del arg783_1
        del buf429
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (8, 448, 14, 14), (87808, 196, 14, 1))
        del arg336_1
        buf432 = buf431; del buf431  # reuse
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf432, arg785_1, arg786_1, arg337_1, arg338_1, 702464, grid=grid(702464), stream=stream0)
        del arg337_1
        del arg338_1
        del arg785_1
        del arg786_1
        buf433 = reinterpret_tensor(buf424, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf424  # reuse
        # Source Nodes: [sp_349], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf432, buf433, 448, 196, grid=grid(448, 196), stream=stream0)
        buf434 = buf423; del buf423  # reuse
        # Source Nodes: [sp_349], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg339_1, buf434, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg339_1
        # Source Nodes: [sp_349], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf433, buf434, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf462 = reinterpret_tensor(buf428, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf428  # reuse
        buf436 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf437 = buf433; del buf433  # reuse
        # Source Nodes: [sp_350, sp_351, sp_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf435, arg788_1, arg789_1, arg340_1, arg341_1, buf432, buf436, buf437, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg340_1
        del arg341_1
        del arg788_1
        del arg789_1
        del buf435
        buf438 = buf434; del buf434  # reuse
        # Source Nodes: [sp_352, sp_353], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg342_1, buf438, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg342_1
        # Source Nodes: [sp_352, sp_353], Original ATen: [aten.add, aten.convolution]
        buf439 = extern_kernels.convolution(buf437, buf438, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf440 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf441 = buf437; del buf437  # reuse
        # Source Nodes: [sp_354, sp_355, sp_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_64.run(buf439, arg791_1, arg792_1, arg343_1, arg344_1, buf432, buf440, buf441, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg343_1
        del arg344_1
        del arg791_1
        del arg792_1
        del buf439
        buf442 = buf438; del buf438  # reuse
        # Source Nodes: [sp_356, sp_357], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg345_1, buf442, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg345_1
        # Source Nodes: [sp_356, sp_357], Original ATen: [aten.add, aten.convolution]
        buf443 = extern_kernels.convolution(buf441, buf442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf444 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf445 = buf441; del buf441  # reuse
        # Source Nodes: [sp_358, sp_359, sp_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf443, arg794_1, arg795_1, arg346_1, arg347_1, buf432, buf444, buf445, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg346_1
        del arg347_1
        del arg794_1
        del arg795_1
        del buf443
        buf446 = buf442; del buf442  # reuse
        # Source Nodes: [sp_360, sp_361], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg348_1, buf446, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg348_1
        # Source Nodes: [sp_360, sp_361], Original ATen: [aten.add, aten.convolution]
        buf447 = extern_kernels.convolution(buf445, buf446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf448 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf449 = buf445; del buf445  # reuse
        # Source Nodes: [sp_362, sp_363, sp_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_66.run(buf447, arg797_1, arg798_1, arg349_1, arg350_1, buf432, buf448, buf449, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg349_1
        del arg350_1
        del arg797_1
        del arg798_1
        del buf447
        buf450 = buf446; del buf446  # reuse
        # Source Nodes: [sp_364, sp_365], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg351_1, buf450, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg351_1
        # Source Nodes: [sp_364, sp_365], Original ATen: [aten.add, aten.convolution]
        buf451 = extern_kernels.convolution(buf449, buf450, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf452 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf453 = buf449; del buf449  # reuse
        # Source Nodes: [sp_366, sp_367, sp_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_67.run(buf451, arg800_1, arg801_1, arg352_1, arg353_1, buf432, buf452, buf453, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg352_1
        del arg353_1
        del arg800_1
        del arg801_1
        del buf451
        buf454 = buf450; del buf450  # reuse
        # Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg354_1, buf454, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg354_1
        # Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        buf455 = extern_kernels.convolution(buf453, buf454, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf456 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf457 = buf453; del buf453  # reuse
        # Source Nodes: [sp_370, sp_371, sp_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_68.run(buf455, arg803_1, arg804_1, arg355_1, arg356_1, buf432, buf456, buf457, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg355_1
        del arg356_1
        del arg803_1
        del arg804_1
        del buf455
        buf458 = buf454; del buf454  # reuse
        # Source Nodes: [sp_372, sp_373], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg357_1, buf458, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg357_1
        # Source Nodes: [sp_372, sp_373], Original ATen: [aten.add, aten.convolution]
        buf459 = extern_kernels.convolution(buf457, buf458, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 56, 14, 14), (10976, 196, 14, 1))
        del buf457
        del buf458
        buf460 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Source Nodes: [sp_374, sp_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf459, arg806_1, arg807_1, arg358_1, arg359_1, buf460, 87808, grid=grid(87808), stream=stream0)
        del arg358_1
        del arg359_1
        del arg806_1
        del arg807_1
        del buf459
        buf461 = reinterpret_tensor(buf462, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf432, buf461, 87808, grid=grid(87808), stream=stream0)
        buf463 = reinterpret_tensor(buf432, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf432  # reuse
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf462, buf463, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf436
        del buf440
        del buf444
        del buf448
        del buf452
        del buf456
        del buf460
        del buf461
        del buf462
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, arg360_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg360_1
        del buf463
        buf465 = buf430; del buf430  # reuse
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_70.run(buf465, buf464, arg809_1, arg810_1, arg361_1, arg362_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg361_1
        del arg362_1
        del arg809_1
        del arg810_1
        del buf464
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 896, 14, 14), (175616, 196, 14, 1))
        del arg363_1
        buf467 = buf466; del buf466  # reuse
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_71.run(buf467, arg812_1, arg813_1, arg364_1, arg365_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg364_1
        del arg365_1
        del arg812_1
        del arg813_1
        buf468 = reinterpret_tensor(buf247, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf247  # reuse
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf467, buf468, 896, 196, grid=grid(896, 196), stream=stream0)
        buf469 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg366_1, buf469, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg366_1
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf468, buf469, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf471 = buf468; del buf468  # reuse
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf467, buf471, 896, 196, grid=grid(896, 196), stream=stream0)
        buf472 = buf469; del buf469  # reuse
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg369_1, buf472, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg369_1
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf471, buf472, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf474 = buf471; del buf471  # reuse
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf467, buf474, 896, 196, grid=grid(896, 196), stream=stream0)
        buf475 = buf472; del buf472  # reuse
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg372_1, buf475, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg372_1
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf474, buf475, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf477 = buf474; del buf474  # reuse
        # Source Nodes: [sp_390], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf467, buf477, 896, 196, grid=grid(896, 196), stream=stream0)
        buf478 = buf475; del buf475  # reuse
        # Source Nodes: [sp_390], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg375_1, buf478, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg375_1
        # Source Nodes: [sp_390], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf477, buf478, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf480 = buf477; del buf477  # reuse
        # Source Nodes: [sp_394], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf467, buf480, 896, 196, grid=grid(896, 196), stream=stream0)
        buf481 = buf478; del buf478  # reuse
        # Source Nodes: [sp_394], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg378_1, buf481, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg378_1
        # Source Nodes: [sp_394], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf480, buf481, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf483 = buf480; del buf480  # reuse
        # Source Nodes: [sp_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_78.run(buf467, buf483, 896, 196, grid=grid(896, 196), stream=stream0)
        buf484 = buf481; del buf481  # reuse
        # Source Nodes: [sp_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg381_1, buf484, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg381_1
        # Source Nodes: [sp_398], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf483, buf484, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf486 = buf483; del buf483  # reuse
        # Source Nodes: [sp_402], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf467, buf486, 896, 196, grid=grid(896, 196), stream=stream0)
        buf487 = buf484; del buf484  # reuse
        # Source Nodes: [sp_402], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg384_1, buf487, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg384_1
        # Source Nodes: [sp_402], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf486, buf487, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (8, 112, 7, 7), (5488, 49, 7, 1))
        del buf486
        buf497 = reinterpret_tensor(buf274, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf274  # reuse
        buf489 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_80.run(buf467, buf489, 43904, grid=grid(43904), stream=stream0)
        del buf467
        buf490 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        # Source Nodes: [sp_379, sp_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf470, arg815_1, arg816_1, arg367_1, arg368_1, buf490, 43904, grid=grid(43904), stream=stream0)
        del arg367_1
        del arg368_1
        del arg815_1
        del arg816_1
        del buf470
        buf491 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        # Source Nodes: [sp_383, sp_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf473, arg818_1, arg819_1, arg370_1, arg371_1, buf491, 43904, grid=grid(43904), stream=stream0)
        del arg370_1
        del arg371_1
        del arg818_1
        del arg819_1
        del buf473
        buf492 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        # Source Nodes: [sp_387, sp_388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf476, arg821_1, arg822_1, arg373_1, arg374_1, buf492, 43904, grid=grid(43904), stream=stream0)
        del arg373_1
        del arg374_1
        del arg821_1
        del arg822_1
        del buf476
        buf493 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        # Source Nodes: [sp_391, sp_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf479, arg824_1, arg825_1, arg376_1, arg377_1, buf493, 43904, grid=grid(43904), stream=stream0)
        del arg376_1
        del arg377_1
        del arg824_1
        del arg825_1
        del buf479
        buf494 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        # Source Nodes: [sp_395, sp_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf482, arg827_1, arg828_1, arg379_1, arg380_1, buf494, 43904, grid=grid(43904), stream=stream0)
        del arg379_1
        del arg380_1
        del arg827_1
        del arg828_1
        del buf482
        buf495 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        # Source Nodes: [sp_399, sp_400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf485, arg830_1, arg831_1, arg382_1, arg383_1, buf495, 43904, grid=grid(43904), stream=stream0)
        del arg382_1
        del arg383_1
        del arg830_1
        del arg831_1
        del buf485
        buf496 = reinterpret_tensor(buf497, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Source Nodes: [sp_403, sp_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf488, arg833_1, arg834_1, arg385_1, arg386_1, buf496, 43904, grid=grid(43904), stream=stream0)
        del arg385_1
        del arg386_1
        del arg833_1
        del arg834_1
        buf498 = reinterpret_tensor(buf103, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf103  # reuse
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf497, buf498, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf489
        del buf490
        del buf491
        del buf492
        del buf493
        del buf494
        del buf495
        del buf496
        del buf497
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, arg387_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg387_1
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf465, arg390_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg390_1
        del buf465
        buf501 = buf499; del buf499  # reuse
        buf502 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_83.run(buf501, arg836_1, arg837_1, arg388_1, arg389_1, buf500, arg839_1, arg840_1, arg391_1, arg392_1, buf502, 16384, 49, grid=grid(16384, 49), stream=stream0)
        del arg388_1
        del arg389_1
        del arg391_1
        del arg392_1
        del arg836_1
        del arg837_1
        del arg839_1
        del arg840_1
        del buf500
        del buf501
        # Source Nodes: [out_112, shortcut_18], Original ATen: [aten.convolution, aten.relu]
        buf503 = extern_kernels.convolution(buf502, arg393_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (8, 896, 7, 7), (43904, 49, 7, 1))
        del arg393_1
        buf504 = buf503; del buf503  # reuse
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf504, arg842_1, arg843_1, arg394_1, arg395_1, 351232, grid=grid(351232), stream=stream0)
        del arg394_1
        del arg395_1
        del arg842_1
        del arg843_1
        buf505 = reinterpret_tensor(buf488, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf488  # reuse
        # Source Nodes: [sp_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf504, buf505, 896, 49, grid=grid(896, 49), stream=stream0)
        buf506 = buf487; del buf487  # reuse
        # Source Nodes: [sp_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg396_1, buf506, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg396_1
        # Source Nodes: [sp_407], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf505, buf506, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf534 = reinterpret_tensor(buf498, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf498  # reuse
        buf508 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf509 = buf505; del buf505  # reuse
        # Source Nodes: [sp_408, sp_409, sp_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_86.run(buf507, arg845_1, arg846_1, arg397_1, arg398_1, buf504, buf508, buf509, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg397_1
        del arg398_1
        del arg845_1
        del arg846_1
        del buf507
        buf510 = buf506; del buf506  # reuse
        # Source Nodes: [sp_410, sp_411], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg399_1, buf510, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg399_1
        # Source Nodes: [sp_410, sp_411], Original ATen: [aten.add, aten.convolution]
        buf511 = extern_kernels.convolution(buf509, buf510, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf512 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        buf513 = buf509; del buf509  # reuse
        # Source Nodes: [sp_412, sp_413, sp_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_87.run(buf511, arg848_1, arg849_1, arg400_1, arg401_1, buf504, buf512, buf513, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg400_1
        del arg401_1
        del arg848_1
        del arg849_1
        del buf511
        buf514 = buf510; del buf510  # reuse
        # Source Nodes: [sp_414, sp_415], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg402_1, buf514, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg402_1
        # Source Nodes: [sp_414, sp_415], Original ATen: [aten.add, aten.convolution]
        buf515 = extern_kernels.convolution(buf513, buf514, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf516 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        buf517 = buf513; del buf513  # reuse
        # Source Nodes: [sp_416, sp_417, sp_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_88.run(buf515, arg851_1, arg852_1, arg403_1, arg404_1, buf504, buf516, buf517, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg403_1
        del arg404_1
        del arg851_1
        del arg852_1
        del buf515
        buf518 = buf514; del buf514  # reuse
        # Source Nodes: [sp_418, sp_419], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg405_1, buf518, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg405_1
        # Source Nodes: [sp_418, sp_419], Original ATen: [aten.add, aten.convolution]
        buf519 = extern_kernels.convolution(buf517, buf518, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf520 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        buf521 = buf517; del buf517  # reuse
        # Source Nodes: [sp_420, sp_421, sp_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_89.run(buf519, arg854_1, arg855_1, arg406_1, arg407_1, buf504, buf520, buf521, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg406_1
        del arg407_1
        del arg854_1
        del arg855_1
        del buf519
        buf522 = buf518; del buf518  # reuse
        # Source Nodes: [sp_422, sp_423], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg408_1, buf522, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg408_1
        # Source Nodes: [sp_422, sp_423], Original ATen: [aten.add, aten.convolution]
        buf523 = extern_kernels.convolution(buf521, buf522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf524 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        buf525 = buf521; del buf521  # reuse
        # Source Nodes: [sp_424, sp_425, sp_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_90.run(buf523, arg857_1, arg858_1, arg409_1, arg410_1, buf504, buf524, buf525, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg409_1
        del arg410_1
        del arg857_1
        del arg858_1
        del buf523
        buf526 = buf522; del buf522  # reuse
        # Source Nodes: [sp_426, sp_427], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg411_1, buf526, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg411_1
        # Source Nodes: [sp_426, sp_427], Original ATen: [aten.add, aten.convolution]
        buf527 = extern_kernels.convolution(buf525, buf526, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf528 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        buf529 = buf525; del buf525  # reuse
        # Source Nodes: [sp_428, sp_429, sp_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_91.run(buf527, arg860_1, arg861_1, arg412_1, arg413_1, buf504, buf528, buf529, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg412_1
        del arg413_1
        del arg860_1
        del arg861_1
        del buf527
        buf530 = buf526; del buf526  # reuse
        # Source Nodes: [sp_430, sp_431], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg414_1, buf530, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg414_1
        # Source Nodes: [sp_430, sp_431], Original ATen: [aten.add, aten.convolution]
        buf531 = extern_kernels.convolution(buf529, buf530, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (8, 112, 7, 7), (5488, 49, 7, 1))
        del buf529
        buf532 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Source Nodes: [sp_432, sp_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf531, arg863_1, arg864_1, arg415_1, arg416_1, buf532, 43904, grid=grid(43904), stream=stream0)
        del arg415_1
        del arg416_1
        del arg863_1
        del arg864_1
        buf533 = reinterpret_tensor(buf534, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_92.run(buf504, buf533, 43904, grid=grid(43904), stream=stream0)
        buf535 = reinterpret_tensor(buf504, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf504  # reuse
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf534, buf535, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf508
        del buf512
        del buf516
        del buf520
        del buf524
        del buf528
        del buf532
        del buf533
        del buf534
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, arg417_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg417_1
        buf537 = buf502; del buf502  # reuse
        # Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_93.run(buf537, buf536, arg866_1, arg867_1, arg418_1, arg419_1, 392, 2048, grid=grid(392, 2048), stream=stream0)
        del arg418_1
        del arg419_1
        del arg866_1
        del arg867_1
        del buf536
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, arg420_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 896, 7, 7), (43904, 49, 7, 1))
        del arg420_1
        buf539 = buf538; del buf538  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf539, arg869_1, arg870_1, arg421_1, arg422_1, 351232, grid=grid(351232), stream=stream0)
        del arg421_1
        del arg422_1
        del arg869_1
        del arg870_1
        buf540 = reinterpret_tensor(buf531, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf531  # reuse
        # Source Nodes: [sp_436], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf539, buf540, 896, 49, grid=grid(896, 49), stream=stream0)
        buf541 = buf530; del buf530  # reuse
        # Source Nodes: [sp_436], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(arg423_1, buf541, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg423_1
        # Source Nodes: [sp_436], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf540, buf541, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf569 = reinterpret_tensor(buf535, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf535  # reuse
        buf543 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf544 = buf540; del buf540  # reuse
        # Source Nodes: [sp_437, sp_438, sp_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_86.run(buf542, arg872_1, arg873_1, arg424_1, arg425_1, buf539, buf543, buf544, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg424_1
        del arg425_1
        del arg872_1
        del arg873_1
        del buf542
        buf545 = buf541; del buf541  # reuse
        # Source Nodes: [sp_439, sp_440], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg426_1, buf545, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg426_1
        # Source Nodes: [sp_439, sp_440], Original ATen: [aten.add, aten.convolution]
        buf546 = extern_kernels.convolution(buf544, buf545, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf547 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        buf548 = buf544; del buf544  # reuse
        # Source Nodes: [sp_441, sp_442, sp_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_87.run(buf546, arg875_1, arg876_1, arg427_1, arg428_1, buf539, buf547, buf548, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg427_1
        del arg428_1
        del arg875_1
        del arg876_1
        del buf546
        buf549 = buf545; del buf545  # reuse
        # Source Nodes: [sp_443, sp_444], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg429_1, buf549, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg429_1
        # Source Nodes: [sp_443, sp_444], Original ATen: [aten.add, aten.convolution]
        buf550 = extern_kernels.convolution(buf548, buf549, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf551 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        buf552 = buf548; del buf548  # reuse
        # Source Nodes: [sp_445, sp_446, sp_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_88.run(buf550, arg878_1, arg879_1, arg430_1, arg431_1, buf539, buf551, buf552, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg430_1
        del arg431_1
        del arg878_1
        del arg879_1
        del buf550
        buf553 = buf549; del buf549  # reuse
        # Source Nodes: [sp_447, sp_448], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg432_1, buf553, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg432_1
        # Source Nodes: [sp_447, sp_448], Original ATen: [aten.add, aten.convolution]
        buf554 = extern_kernels.convolution(buf552, buf553, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf555 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        buf556 = buf552; del buf552  # reuse
        # Source Nodes: [sp_449, sp_450, sp_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_89.run(buf554, arg881_1, arg882_1, arg433_1, arg434_1, buf539, buf555, buf556, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg433_1
        del arg434_1
        del arg881_1
        del arg882_1
        del buf554
        buf557 = buf553; del buf553  # reuse
        # Source Nodes: [sp_451, sp_452], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg435_1, buf557, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg435_1
        # Source Nodes: [sp_451, sp_452], Original ATen: [aten.add, aten.convolution]
        buf558 = extern_kernels.convolution(buf556, buf557, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf559 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        buf560 = buf556; del buf556  # reuse
        # Source Nodes: [sp_453, sp_454, sp_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_90.run(buf558, arg884_1, arg885_1, arg436_1, arg437_1, buf539, buf559, buf560, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg436_1
        del arg437_1
        del arg884_1
        del arg885_1
        del buf558
        buf561 = buf557; del buf557  # reuse
        # Source Nodes: [sp_455, sp_456], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg438_1, buf561, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg438_1
        # Source Nodes: [sp_455, sp_456], Original ATen: [aten.add, aten.convolution]
        buf562 = extern_kernels.convolution(buf560, buf561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf563 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        buf564 = buf560; del buf560  # reuse
        # Source Nodes: [sp_457, sp_458, sp_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_91.run(buf562, arg887_1, arg888_1, arg439_1, arg440_1, buf539, buf563, buf564, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg439_1
        del arg440_1
        del arg887_1
        del arg888_1
        del buf562
        buf565 = buf561; del buf561  # reuse
        # Source Nodes: [sp_459, sp_460], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_73.run(arg441_1, buf565, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg441_1
        # Source Nodes: [sp_459, sp_460], Original ATen: [aten.add, aten.convolution]
        buf566 = extern_kernels.convolution(buf564, buf565, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 112, 7, 7), (5488, 49, 7, 1))
        del buf564
        del buf565
        buf567 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Source Nodes: [sp_461, sp_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_81.run(buf566, arg890_1, arg891_1, arg442_1, arg443_1, buf567, 43904, grid=grid(43904), stream=stream0)
        del arg442_1
        del arg443_1
        del arg890_1
        del arg891_1
        del buf566
        buf568 = reinterpret_tensor(buf569, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_92.run(buf539, buf568, 43904, grid=grid(43904), stream=stream0)
        buf570 = reinterpret_tensor(buf539, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf539  # reuse
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf569, buf570, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf543
        del buf547
        del buf551
        del buf555
        del buf559
        del buf563
        del buf567
        del buf568
        del buf569
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, arg444_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg444_1
        del buf570
        buf572 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf573 = reinterpret_tensor(buf572, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf572  # reuse
        # Source Nodes: [out_125, out_126, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_94.run(buf573, buf571, arg893_1, arg894_1, arg445_1, arg446_1, buf537, 16384, 49, grid=grid(16384), stream=stream0)
        del arg445_1
        del arg446_1
        del arg893_1
        del arg894_1
        del buf537
        del buf571
        buf574 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg448_1, reinterpret_tensor(buf573, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg447_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf574)
        del arg447_1
        del arg448_1
        return (buf574, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((112, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((224, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((448, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((896, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg452_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg455_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg458_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg461_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg464_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg467_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg470_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg473_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg476_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg479_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg482_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg485_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg488_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg491_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg494_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg497_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg500_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg503_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg506_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg509_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg512_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg518_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg521_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg524_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg527_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg530_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg533_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg536_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg539_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg542_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg545_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg548_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg551_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg554_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg557_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg560_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg563_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg566_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg569_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg572_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg575_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg578_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg581_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg584_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg587_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg590_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg593_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg596_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg599_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg602_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg605_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg608_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg611_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg614_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg617_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg620_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg623_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg626_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg629_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg632_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg635_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg638_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg641_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg644_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg647_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg650_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg653_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg656_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg659_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg662_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg665_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg668_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg671_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg674_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg677_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg680_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg683_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg686_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg689_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg692_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg695_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg698_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg701_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg704_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg707_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg710_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg713_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg716_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg719_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg722_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg725_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg728_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg731_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg734_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg737_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg740_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg743_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg746_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg749_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg752_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg755_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg758_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg761_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg764_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg767_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg770_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg773_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg776_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg779_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg782_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg785_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg788_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg791_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg794_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg797_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg800_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg803_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg806_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg809_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg812_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg815_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg818_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg821_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg824_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg827_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg830_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg833_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg836_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg839_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg842_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg845_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg848_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg851_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg854_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg857_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg860_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg863_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg866_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg869_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg872_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg875_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg878_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg881_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg884_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg887_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg890_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg893_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg896_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net50_14w_8s', benchmark_compiled_module)
