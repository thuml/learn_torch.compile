
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyej23vyw5763kmux2dywgfrfqsdpwbf6bflpsog6h63cyycpqw.py
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
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rp/crpyejihhghoq2yx2xrqpkpbbluhixyr7ir7p4imx6ilwsdv7bcr.py
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
    size_hints=[16, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqqpfwbujsjlbk745afxirvfnmmtiskk6d437vjnlabuawnnhha.py
# Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
# l__mod___conv1_0 => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5xime2hvceg4yq4jk3ndn5mqjyet2lbbfm2wza4ir2ckbjjmwg.py
# Source Nodes: [l__mod___conv1_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___conv1_1 => add_1, mul_1, mul_2, sub
# x => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzoiyt4q7wwylc3yiyybmgeoucgfne75yqsgtuufa2xvlpr7s4d.py
# Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
# x_1 => getitem, getitem_1
triton_poi_fused_max_pool2d_with_indices_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1344) % 56
    x1 = (xindex // 24) % 56
    x0 = xindex % 24
    x5 = (xindex // 1344)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2712) + x0 + (48*x1) + (5376*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-2688) + x0 + (48*x1) + (5376*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-2664) + x0 + (48*x1) + (5376*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-24) + x0 + (48*x1) + (5376*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x0 + (48*x1) + (5376*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (24 + x0 + (48*x1) + (5376*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (2664 + x0 + (48*x1) + (5376*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (2688 + x0 + (48*x1) + (5376*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (2712 + x0 + (48*x1) + (5376*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-112) + (2*x1) + (224*x2)
    tmp72 = (-113) + (2*x1) + (224*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-111) + (2*x1) + (224*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x1) + (224*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x1) + (224*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x1) + (224*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 111 + (2*x1) + (224*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 112 + (2*x1) + (224*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 113 + (2*x1) + (224*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2mpnqjynmuvfynjbgqf4slkhoiyunrv4ljpavgtqoh7z665en3w.py
# Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage2___0___branch1_0 => convolution_1
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (18816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgv23kxj5zi2tia7hjki2xnsnqeaj25bz6koe55jhkcgckyz4fl3.py
# Source Nodes: [getattr_l__mod___stage2___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage2___0___branch1_1 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwwqkzxxncipmory2n2c4uwi7dla7jyg46h2xgolk3tltpa2exx.py
# Source Nodes: [getattr_l__mod___stage2___0___branch1_2, getattr_l__mod___stage2___0___branch1_3, getattr_l__mod___stage2___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage2___0___branch1_2 => convolution_2
# getattr_l__mod___stage2___0___branch1_3 => add_5, mul_7, mul_8, sub_2
# getattr_l__mod___stage2___0___branch1_4 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (784*y0) + (90944*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (58*x2) + (45472*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyicvzgfivohkjo2smkl76vpdhzvqlq23d7lsf67njq66gve2nh.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage2___0___branch2_0 => convolution_3
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (58*x2) + (181888*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwuur4vwbxtdz3y2cjqapsm6ynbkl4c6mhs7v66np7ksfynfdwz.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage2___0___branch2_1 => add_7, mul_10, mul_11, sub_3
# getattr_l__mod___stage2___0___branch2_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 727552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpm6ykhjkvl7psvcbrbbxrpeswqmu5da7k6l4tsazh2zzxgnmk6.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten.convolution]
# getattr_l__mod___stage2___0___branch2_3 => convolution_4
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcqeiic2odnro47furfaqmbvhc6l7vvur47bx7n7rj3aiv7cgsv.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage2___0___branch2_4 => add_9, mul_13, mul_14, sub_4
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhmef6anooobt5gskot3vklltybtkknb7w2vmffx2kdeu4frrw3.py
# Source Nodes: [x_3], Original ATen: [aten.clone]
# x_3 => clone
triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 784
    x1 = (xindex // 784) % 2
    x2 = (xindex // 1568) % 58
    x3 = (xindex // 90944)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*x2) + (45472*x1) + (90944*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskw2fjv635azzmqxm7ya444kweofj72oc3em7zc3jcjrrit43mz.py
# Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage2___1___branch2_0 => convolution_6
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (45472 + x2 + (784*y0) + (90944*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbuwtwzjjnffqwslqj23byzffpnuynbrcqhbov67gpc23wtwc7bh.py
# Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage2___1___branch2_1 => add_13, mul_19, mul_20, sub_6
# getattr_l__mod___stage2___1___branch2_2 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c543hswmkhlrubiev3etuaaj5imjn4xeveqcu33jvfwfibvzs5q3.py
# Source Nodes: [getattr_l__mod___stage2___1___branch2_6, getattr_l__mod___stage2___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage2___1___branch2_6 => add_17, mul_25, mul_26, sub_8
# getattr_l__mod___stage2___1___branch2_7 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugaagggsodvzgovx5rzszv72gzmu7rumwichjteztgpxt6pjpmn.py
# Source Nodes: [x_6], Original ATen: [aten.clone]
# x_6 => clone_1
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y4 = yindex % 116
    x3 = xindex
    y5 = yindex
    y2 = (yindex // 116)
    y0 = yindex % 58
    y1 = (yindex // 58) % 2
    tmp0 = y4
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 58, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (784*y5)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 116, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-58) + y4 + (58*x3) + (45472*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x3 + (784*y1) + (1568*y0) + (90944*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l3fcl727mj3kjeog6qufsve6okpbbrmi42aizwxh4fh7pfw6mr.py
# Source Nodes: [x_12, x_14], Original ATen: [aten.clone, aten.view]
# x_12 => clone_3
# x_14 => view_7
triton_poi_fused_clone_view_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12992
    xnumel = 28
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 116
    x3 = xindex
    y1 = (yindex // 116) % 28
    y2 = (yindex // 3248)
    y4 = (yindex // 116)
    tmp0 = (58*(y0 % 2)) + (y0 // 2)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 58, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (28*y1) + (784*((y0 // 2) % 2)) + (1568*((((58*(y0 % 2)) + (y0 // 2)) // 2) % 58)) + (90944*y2)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 116, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-58) + (58*x3) + (58*(y0 % 2)) + (1624*y4) + (y0 // 2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (116*x3) + (3248*y4)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdprk3dzlzmgmaid63lsdg56gwp7rwybzhyiyv3vk74vjbqej6sq.py
# Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage3___0___branch1_0 => convolution_15
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwehhxlcgcxkx2wssrgxctsjw7mmv4nctiv2wsgfidxp7dzdute.py
# Source Nodes: [getattr_l__mod___stage3___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage3___0___branch1_1 => add_31, mul_46, mul_47, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 116
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybatnzx53bzkqh32us5lmfcp26zglnb4op6rlylwptpqrkrfxej.py
# Source Nodes: [getattr_l__mod___stage3___0___branch1_2, getattr_l__mod___stage3___0___branch1_3, getattr_l__mod___stage3___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage3___0___branch1_2 => convolution_16
# getattr_l__mod___stage3___0___branch1_3 => add_33, mul_49, mul_50, sub_16
# getattr_l__mod___stage3___0___branch1_4 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (196*y0) + (45472*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (116*x2) + (22736*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7k2y4pbmq5yheidw3spbxj6tzf23o2v2qctykfopfy63iwxc47.py
# Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage3___0___branch2_0 => convolution_17
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (116*x2) + (90944*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/cziwbnrvaj2jklpyucqyi2hs7owb576gvpge6fpfyaxfzmuakpfn.py
# Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage3___0___branch2_1 => add_35, mul_52, mul_53, sub_17
# getattr_l__mod___stage3___0___branch2_2 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 116
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csg53kpxfq3efwppxyhhewwnyeqzd33eulvewgrriaqh2u5chdrf.py
# Source Nodes: [x_16], Original ATen: [aten.clone]
# x_16 => clone_4
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 196
    x1 = (xindex // 196) % 2
    x2 = (xindex // 392) % 116
    x3 = (xindex // 45472)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*x2) + (22736*x1) + (45472*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4crg2xpxf2alpsyjylm3u33ml4mxhtevi3ugex4ccjiua4qyeg.py
# Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage3___1___branch2_0 => convolution_20
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (22736 + x2 + (196*y0) + (45472*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6td7lkgvha3ibr55lpusyl2jyocbk6avjlr673b5qcllldgcbim.py
# Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage3___1___branch2_1 => add_41, mul_61, mul_62, sub_20
# getattr_l__mod___stage3___1___branch2_2 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 116
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gniky7k3rqwslost6q6xl7f6xrbs6tdskfdh5rq7ydmtkxpmfb.py
# Source Nodes: [getattr_l__mod___stage3___1___branch2_6, getattr_l__mod___stage3___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage3___1___branch2_6 => add_45, mul_67, mul_68, sub_22
# getattr_l__mod___stage3___1___branch2_7 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 116
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2htzf75tv6czke5fn5qwrym6jananzk6ydbxnk4nuuz7lbnn2c6.py
# Source Nodes: [x_19], Original ATen: [aten.clone]
# x_19 => clone_5
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y4 = yindex % 232
    x3 = xindex
    y5 = yindex
    y2 = (yindex // 232)
    y0 = yindex % 116
    y1 = (yindex // 116) % 2
    tmp0 = y4
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (196*y5)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 232, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-116) + y4 + (116*x3) + (22736*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x3 + (196*y1) + (392*y0) + (45472*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl76cv6tlif5zxemxedx5xjtqx3xkgvqff2z5wupmnmielrmeave.py
# Source Nodes: [x_37, x_39], Original ATen: [aten.clone, aten.view]
# x_37 => clone_11
# x_39 => view_23
triton_poi_fused_clone_view_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12992
    xnumel = 14
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 232
    x3 = xindex
    y1 = (yindex // 232) % 14
    y2 = (yindex // 3248)
    y4 = (yindex // 232)
    tmp0 = (116*(y0 % 2)) + (y0 // 2)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (14*y1) + (196*((y0 // 2) % 2)) + (392*((((116*(y0 % 2)) + (y0 // 2)) // 2) % 116)) + (45472*y2)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 232, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-116) + (116*x3) + (116*(y0 % 2)) + (1624*y4) + (y0 // 2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (232*x3) + (3248*y4)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvufmoetemaa3hbr2yfjfil45niolg7hr2mefd57eanz5e432fqr.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage4___0___branch1_0 => convolution_41
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gb2oggi4552bcklbr2v4v6gfxwreiwvetulr5uqsqm2nwzo7jl.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage4___0___branch1_1 => add_83, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 232
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjf4aq6kzytcphylkiiqy4hooa3uyc62qheueoshzgahz5ybyags.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_2, getattr_l__mod___stage4___0___branch1_3, getattr_l__mod___stage4___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage4___0___branch1_2 => convolution_42
# getattr_l__mod___stage4___0___branch1_3 => add_85, mul_127, mul_128, sub_42
# getattr_l__mod___stage4___0___branch1_4 => relu_27
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (49*y0) + (22736*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (232*x2) + (11368*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qjml7lpvbatispftvlf5lejt6awhyrbyd7tqbpok6l4ez6xrnu.py
# Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage4___0___branch2_0 => convolution_43
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (232*x2) + (45472*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciac2zw5b64z252qrmuolbtlzfxgyznpr6dkck2tvpeyssvy6so7.py
# Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage4___0___branch2_1 => add_87, mul_130, mul_131, sub_43
# getattr_l__mod___stage4___0___branch2_2 => relu_28
triton_poi_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 232
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6e6t5rmrjz46tnrfg7xp43m6zhvya2w2lamni5e72mrfp2diqd.py
# Source Nodes: [getattr_l__mod___stage4___0___branch2_5, getattr_l__mod___stage4___0___branch2_6, getattr_l__mod___stage4___0___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage4___0___branch2_5 => convolution_45
# getattr_l__mod___stage4___0___branch2_6 => add_91, mul_136, mul_137, sub_45
# getattr_l__mod___stage4___0___branch2_7 => relu_29
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (49*y0) + (22736*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (232*x2) + (11368*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5ktw5lf6wcww2kpvkkbvk72exlwiinu3cvder4uhogysx7xnq4w.py
# Source Nodes: [x_41], Original ATen: [aten.clone]
# x_41 => clone_12
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 49
    x1 = (xindex // 49) % 2
    x2 = (xindex // 98) % 232
    x3 = (xindex // 22736)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*x2) + (11368*x1) + (22736*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxnyznsdu72pe3rducjnw6c6e4jbejeuekb2wtaghy4pnoltjiv.py
# Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage4___1___branch2_0 => convolution_46
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (11368 + x2 + (49*y0) + (22736*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/css5y7nu4fhbmtxck3myk7ahsmmhvs7puhitcc73gsmcjf2bg6xl.py
# Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage4___1___branch2_1 => add_93, mul_139, mul_140, sub_46
# getattr_l__mod___stage4___1___branch2_2 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 232
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuy62z2mlqrixddn4jqy2da4cfivtexpwh5vhfpqm4knm2hlhzi2.py
# Source Nodes: [getattr_l__mod___stage4___1___branch2_6, getattr_l__mod___stage4___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# getattr_l__mod___stage4___1___branch2_6 => add_97, mul_145, mul_146, sub_48
# getattr_l__mod___stage4___1___branch2_7 => relu_31
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 232
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsejk6tg4qxssz4iz7huemjofyuafgasqewumq2wmsoxi2yorhn.py
# Source Nodes: [x_44], Original ATen: [aten.clone]
# x_44 => clone_13
triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1856
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y4 = yindex % 464
    x3 = xindex
    y5 = yindex
    y2 = (yindex // 464)
    y0 = yindex % 232
    y1 = (yindex // 232) % 2
    tmp0 = y4
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (49*y5)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 464, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-232) + y4 + (232*x3) + (11368*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x3 + (49*y1) + (98*y0) + (22736*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c463mfzpkh6e3yojhxidykpzuqcgkkmayqfl4xedneg7yql5jsrn.py
# Source Nodes: [x_50, x_52], Original ATen: [aten.clone, aten.view]
# x_50 => clone_15
# x_52 => view_31
triton_poi_fused_clone_view_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12992
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 464
    x3 = xindex
    y1 = (yindex // 464) % 7
    y2 = (yindex // 3248)
    y4 = (yindex // 464)
    tmp0 = (232*(y0 % 2)) + (y0 // 2)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (7*y1) + (49*((y0 // 2) % 2)) + (98*((((232*(y0 % 2)) + (y0 // 2)) // 2) % 232)) + (22736*y2)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 464, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-232) + (232*x3) + (232*(y0 % 2)) + (1624*y4) + (y0 // 2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (464*x3) + (3248*y4)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6ncr5iakxar3czzdhv73ssuv53nht5xql6bksmkrrcbbzwwvkw.py
# Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
# l__mod___conv5_0 => convolution_55
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7i/c7iy7hm3tofub4k4qrwiycwron73q2phvswxpskv4qvddrt7masf.py
# Source Nodes: [l__mod___conv5_1, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# l__mod___conv5_1 => add_111, mul_166, mul_167, sub_55
# x_53 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr1 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohkezvcome6yn63shi2jxwzwjdkfaysq42wdfltpvcledllxkj2.py
# Source Nodes: [x_54], Original ATen: [aten.mean]
# x_54 => mean
triton_per_fused_mean_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_8, (58, ), (1, ))
    assert_size_stride(primals_9, (58, ), (1, ))
    assert_size_stride(primals_10, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_11, (58, ), (1, ))
    assert_size_stride(primals_12, (58, ), (1, ))
    assert_size_stride(primals_13, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (58, ), (1, ))
    assert_size_stride(primals_15, (58, ), (1, ))
    assert_size_stride(primals_16, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_17, (58, ), (1, ))
    assert_size_stride(primals_18, (58, ), (1, ))
    assert_size_stride(primals_19, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_20, (58, ), (1, ))
    assert_size_stride(primals_21, (58, ), (1, ))
    assert_size_stride(primals_22, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (58, ), (1, ))
    assert_size_stride(primals_24, (58, ), (1, ))
    assert_size_stride(primals_25, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_26, (58, ), (1, ))
    assert_size_stride(primals_27, (58, ), (1, ))
    assert_size_stride(primals_28, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_29, (58, ), (1, ))
    assert_size_stride(primals_30, (58, ), (1, ))
    assert_size_stride(primals_31, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (58, ), (1, ))
    assert_size_stride(primals_33, (58, ), (1, ))
    assert_size_stride(primals_34, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_35, (58, ), (1, ))
    assert_size_stride(primals_36, (58, ), (1, ))
    assert_size_stride(primals_37, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_38, (58, ), (1, ))
    assert_size_stride(primals_39, (58, ), (1, ))
    assert_size_stride(primals_40, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (58, ), (1, ))
    assert_size_stride(primals_42, (58, ), (1, ))
    assert_size_stride(primals_43, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_44, (58, ), (1, ))
    assert_size_stride(primals_45, (58, ), (1, ))
    assert_size_stride(primals_46, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (116, ), (1, ))
    assert_size_stride(primals_48, (116, ), (1, ))
    assert_size_stride(primals_49, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_50, (116, ), (1, ))
    assert_size_stride(primals_51, (116, ), (1, ))
    assert_size_stride(primals_52, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_53, (116, ), (1, ))
    assert_size_stride(primals_54, (116, ), (1, ))
    assert_size_stride(primals_55, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (116, ), (1, ))
    assert_size_stride(primals_57, (116, ), (1, ))
    assert_size_stride(primals_58, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_59, (116, ), (1, ))
    assert_size_stride(primals_60, (116, ), (1, ))
    assert_size_stride(primals_61, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_62, (116, ), (1, ))
    assert_size_stride(primals_63, (116, ), (1, ))
    assert_size_stride(primals_64, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (116, ), (1, ))
    assert_size_stride(primals_66, (116, ), (1, ))
    assert_size_stride(primals_67, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_68, (116, ), (1, ))
    assert_size_stride(primals_69, (116, ), (1, ))
    assert_size_stride(primals_70, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_71, (116, ), (1, ))
    assert_size_stride(primals_72, (116, ), (1, ))
    assert_size_stride(primals_73, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (116, ), (1, ))
    assert_size_stride(primals_75, (116, ), (1, ))
    assert_size_stride(primals_76, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_77, (116, ), (1, ))
    assert_size_stride(primals_78, (116, ), (1, ))
    assert_size_stride(primals_79, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_80, (116, ), (1, ))
    assert_size_stride(primals_81, (116, ), (1, ))
    assert_size_stride(primals_82, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (116, ), (1, ))
    assert_size_stride(primals_84, (116, ), (1, ))
    assert_size_stride(primals_85, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_86, (116, ), (1, ))
    assert_size_stride(primals_87, (116, ), (1, ))
    assert_size_stride(primals_88, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_89, (116, ), (1, ))
    assert_size_stride(primals_90, (116, ), (1, ))
    assert_size_stride(primals_91, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (116, ), (1, ))
    assert_size_stride(primals_93, (116, ), (1, ))
    assert_size_stride(primals_94, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_95, (116, ), (1, ))
    assert_size_stride(primals_96, (116, ), (1, ))
    assert_size_stride(primals_97, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_98, (116, ), (1, ))
    assert_size_stride(primals_99, (116, ), (1, ))
    assert_size_stride(primals_100, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (116, ), (1, ))
    assert_size_stride(primals_102, (116, ), (1, ))
    assert_size_stride(primals_103, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_104, (116, ), (1, ))
    assert_size_stride(primals_105, (116, ), (1, ))
    assert_size_stride(primals_106, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_107, (116, ), (1, ))
    assert_size_stride(primals_108, (116, ), (1, ))
    assert_size_stride(primals_109, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (116, ), (1, ))
    assert_size_stride(primals_111, (116, ), (1, ))
    assert_size_stride(primals_112, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_113, (116, ), (1, ))
    assert_size_stride(primals_114, (116, ), (1, ))
    assert_size_stride(primals_115, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_116, (116, ), (1, ))
    assert_size_stride(primals_117, (116, ), (1, ))
    assert_size_stride(primals_118, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (116, ), (1, ))
    assert_size_stride(primals_120, (116, ), (1, ))
    assert_size_stride(primals_121, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_122, (116, ), (1, ))
    assert_size_stride(primals_123, (116, ), (1, ))
    assert_size_stride(primals_124, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (232, ), (1, ))
    assert_size_stride(primals_126, (232, ), (1, ))
    assert_size_stride(primals_127, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_128, (232, ), (1, ))
    assert_size_stride(primals_129, (232, ), (1, ))
    assert_size_stride(primals_130, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_131, (232, ), (1, ))
    assert_size_stride(primals_132, (232, ), (1, ))
    assert_size_stride(primals_133, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (232, ), (1, ))
    assert_size_stride(primals_135, (232, ), (1, ))
    assert_size_stride(primals_136, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_137, (232, ), (1, ))
    assert_size_stride(primals_138, (232, ), (1, ))
    assert_size_stride(primals_139, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_140, (232, ), (1, ))
    assert_size_stride(primals_141, (232, ), (1, ))
    assert_size_stride(primals_142, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (232, ), (1, ))
    assert_size_stride(primals_144, (232, ), (1, ))
    assert_size_stride(primals_145, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_146, (232, ), (1, ))
    assert_size_stride(primals_147, (232, ), (1, ))
    assert_size_stride(primals_148, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_149, (232, ), (1, ))
    assert_size_stride(primals_150, (232, ), (1, ))
    assert_size_stride(primals_151, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (232, ), (1, ))
    assert_size_stride(primals_153, (232, ), (1, ))
    assert_size_stride(primals_154, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_155, (232, ), (1, ))
    assert_size_stride(primals_156, (232, ), (1, ))
    assert_size_stride(primals_157, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_158, (232, ), (1, ))
    assert_size_stride(primals_159, (232, ), (1, ))
    assert_size_stride(primals_160, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (232, ), (1, ))
    assert_size_stride(primals_162, (232, ), (1, ))
    assert_size_stride(primals_163, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_164, (232, ), (1, ))
    assert_size_stride(primals_165, (232, ), (1, ))
    assert_size_stride(primals_166, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1000, 1024), (1024, 1))
    assert_size_stride(primals_170, (1000, ), (1, ))
    assert_size_stride(primals_171, (24, ), (1, ))
    assert_size_stride(primals_172, (24, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (58, ), (1, ))
    assert_size_stride(primals_178, (58, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (58, ), (1, ))
    assert_size_stride(primals_181, (58, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (58, ), (1, ))
    assert_size_stride(primals_184, (58, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (58, ), (1, ))
    assert_size_stride(primals_187, (58, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (58, ), (1, ))
    assert_size_stride(primals_190, (58, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (58, ), (1, ))
    assert_size_stride(primals_193, (58, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (58, ), (1, ))
    assert_size_stride(primals_196, (58, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (58, ), (1, ))
    assert_size_stride(primals_199, (58, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (58, ), (1, ))
    assert_size_stride(primals_202, (58, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (58, ), (1, ))
    assert_size_stride(primals_205, (58, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (58, ), (1, ))
    assert_size_stride(primals_208, (58, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (58, ), (1, ))
    assert_size_stride(primals_211, (58, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (58, ), (1, ))
    assert_size_stride(primals_214, (58, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (116, ), (1, ))
    assert_size_stride(primals_217, (116, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (116, ), (1, ))
    assert_size_stride(primals_220, (116, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (116, ), (1, ))
    assert_size_stride(primals_223, (116, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (116, ), (1, ))
    assert_size_stride(primals_226, (116, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (116, ), (1, ))
    assert_size_stride(primals_229, (116, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (116, ), (1, ))
    assert_size_stride(primals_232, (116, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (116, ), (1, ))
    assert_size_stride(primals_235, (116, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (116, ), (1, ))
    assert_size_stride(primals_238, (116, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (116, ), (1, ))
    assert_size_stride(primals_241, (116, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (116, ), (1, ))
    assert_size_stride(primals_244, (116, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (116, ), (1, ))
    assert_size_stride(primals_247, (116, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (116, ), (1, ))
    assert_size_stride(primals_250, (116, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (116, ), (1, ))
    assert_size_stride(primals_253, (116, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (116, ), (1, ))
    assert_size_stride(primals_256, (116, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (116, ), (1, ))
    assert_size_stride(primals_259, (116, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (116, ), (1, ))
    assert_size_stride(primals_262, (116, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (116, ), (1, ))
    assert_size_stride(primals_265, (116, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (116, ), (1, ))
    assert_size_stride(primals_268, (116, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (116, ), (1, ))
    assert_size_stride(primals_271, (116, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (116, ), (1, ))
    assert_size_stride(primals_274, (116, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (116, ), (1, ))
    assert_size_stride(primals_277, (116, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (116, ), (1, ))
    assert_size_stride(primals_280, (116, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (116, ), (1, ))
    assert_size_stride(primals_283, (116, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (116, ), (1, ))
    assert_size_stride(primals_286, (116, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (116, ), (1, ))
    assert_size_stride(primals_289, (116, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (116, ), (1, ))
    assert_size_stride(primals_292, (116, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (232, ), (1, ))
    assert_size_stride(primals_295, (232, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (232, ), (1, ))
    assert_size_stride(primals_298, (232, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (232, ), (1, ))
    assert_size_stride(primals_301, (232, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (232, ), (1, ))
    assert_size_stride(primals_304, (232, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (232, ), (1, ))
    assert_size_stride(primals_307, (232, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (232, ), (1, ))
    assert_size_stride(primals_310, (232, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (232, ), (1, ))
    assert_size_stride(primals_313, (232, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (232, ), (1, ))
    assert_size_stride(primals_316, (232, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (232, ), (1, ))
    assert_size_stride(primals_319, (232, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (232, ), (1, ))
    assert_size_stride(primals_322, (232, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (232, ), (1, ))
    assert_size_stride(primals_325, (232, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (232, ), (1, ))
    assert_size_stride(primals_328, (232, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (232, ), (1, ))
    assert_size_stride(primals_331, (232, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (232, ), (1, ))
    assert_size_stride(primals_334, (232, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 72, 9, grid=grid(72, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_339, buf1, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_339
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 24, 112, 112), (301056, 12544, 112, 1))
        buf3 = empty_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 96, 12544, grid=grid(96, 12544), stream=stream0)
        buf4 = reinterpret_tensor(buf2, (4, 24, 112, 112), (301056, 1, 2688, 24), 0); del buf2  # reuse
        # Source Nodes: [l__mod___conv1_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf3, primals_171, primals_172, primals_2, primals_3, buf4, 1204224, grid=grid(1204224), stream=stream0)
        del primals_3
        buf5 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_4.run(buf4, buf5, buf6, 301056, grid=grid(301056), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf5, primals_4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf7, (4, 24, 28, 28), (18816, 784, 28, 1))
        buf8 = empty_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf7, buf8, 96, 784, grid=grid(96, 784), stream=stream0)
        buf9 = reinterpret_tensor(buf7, (4, 24, 28, 28), (18816, 1, 672, 24), 0); del buf7  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf8, primals_174, primals_175, primals_5, primals_6, buf9, 75264, grid=grid(75264), stream=stream0)
        del primals_6
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_2], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf11 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        buf22 = empty((4, 116, 28, 28), device='cuda', dtype=torch.float32)
        buf12 = reinterpret_tensor(buf22, (4, 58, 28, 28), (90944, 784, 28, 1), 0)  # alias
        buf226 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_2, getattr_l__mod___stage2___0___branch1_3, getattr_l__mod___stage2___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_7.run(buf10, primals_177, primals_178, primals_8, primals_9, buf11, buf12, buf226, 232, 784, grid=grid(232, 784), stream=stream0)
        del primals_9
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf5, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 58, 56, 56), (181888, 3136, 56, 1))
        buf14 = empty_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf13, buf14, 232, 3136, grid=grid(232, 3136), stream=stream0)
        buf15 = reinterpret_tensor(buf13, (4, 58, 56, 56), (181888, 1, 3248, 58), 0); del buf13  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf14, primals_180, primals_181, primals_11, primals_12, buf15, 727552, grid=grid(727552), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf16, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf17 = reinterpret_tensor(buf10, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf10  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf16, buf17, 232, 784, grid=grid(232, 784), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf16  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf17, primals_183, primals_184, primals_14, primals_15, buf18, 181888, grid=grid(181888), stream=stream0)
        del primals_15
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_5], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf20 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf22, (4, 58, 28, 28), (90944, 784, 28, 1), 45472)  # alias
        buf225 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_5, getattr_l__mod___stage2___0___branch2_6, getattr_l__mod___stage2___0___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_7.run(buf19, primals_186, primals_187, primals_17, primals_18, buf20, buf21, buf225, 232, 784, grid=grid(232, 784), stream=stream0)
        del primals_18
        buf23 = empty((4, 58, 2, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf22, buf23, 363776, grid=grid(363776), stream=stream0)
        del buf12
        del buf21
        buf24 = reinterpret_tensor(buf19, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf19  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf23, buf24, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf25, buf26, 232, 784, grid=grid(232, 784), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf25  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf26, primals_189, primals_190, primals_20, primals_21, buf27, 181888, grid=grid(181888), stream=stream0)
        del primals_21
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf28, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf29 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf28, buf29, 232, 784, grid=grid(232, 784), stream=stream0)
        buf30 = reinterpret_tensor(buf28, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf28  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf29, primals_192, primals_193, primals_23, primals_24, buf30, 181888, grid=grid(181888), stream=stream0)
        del primals_24
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_5], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf32 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf31, buf32, 232, 784, grid=grid(232, 784), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf31  # reuse
        buf224 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_6, getattr_l__mod___stage2___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_15.run(buf32, primals_195, primals_196, primals_26, primals_27, buf33, buf224, 181888, grid=grid(181888), stream=stream0)
        del primals_27
        buf34 = reinterpret_tensor(buf22, (4, 58, 2, 28, 28), (90944, 1568, 784, 28, 1), 0); del buf22  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf23, buf33, buf34, 464, 784, grid=grid(464, 784), stream=stream0)
        buf35 = buf33; del buf33  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf34, buf35, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf37 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf36, buf37, 232, 784, grid=grid(232, 784), stream=stream0)
        buf38 = reinterpret_tensor(buf36, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf36  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_1, getattr_l__mod___stage2___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf37, primals_198, primals_199, primals_29, primals_30, buf38, 181888, grid=grid(181888), stream=stream0)
        del primals_30
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_3], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf39, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf40 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf39, buf40, 232, 784, grid=grid(232, 784), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf39  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf40, primals_201, primals_202, primals_32, primals_33, buf41, 181888, grid=grid(181888), stream=stream0)
        del primals_33
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_5], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf43 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf42, buf43, 232, 784, grid=grid(232, 784), stream=stream0)
        buf44 = reinterpret_tensor(buf42, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf42  # reuse
        buf223 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_6, getattr_l__mod___stage2___2___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_15.run(buf43, primals_204, primals_205, primals_35, primals_36, buf44, buf223, 181888, grid=grid(181888), stream=stream0)
        del primals_36
        buf45 = empty((4, 58, 2, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf34, buf44, buf45, 464, 784, grid=grid(464, 784), stream=stream0)
        buf46 = buf44; del buf44  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf45, buf46, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf48 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf47, buf48, 232, 784, grid=grid(232, 784), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf47  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_1, getattr_l__mod___stage2___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf48, primals_207, primals_208, primals_38, primals_39, buf49, 181888, grid=grid(181888), stream=stream0)
        del primals_39
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_3], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf50, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf51 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf50, buf51, 232, 784, grid=grid(232, 784), stream=stream0)
        buf52 = reinterpret_tensor(buf50, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf50  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf51, primals_210, primals_211, primals_41, primals_42, buf52, 181888, grid=grid(181888), stream=stream0)
        del primals_42
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_5], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 58, 28, 28), (45472, 784, 28, 1))
        buf54 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf53, buf54, 232, 784, grid=grid(232, 784), stream=stream0)
        buf55 = reinterpret_tensor(buf53, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf53  # reuse
        buf222 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_6, getattr_l__mod___stage2___3___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_15.run(buf54, primals_213, primals_214, primals_44, primals_45, buf55, buf222, 181888, grid=grid(181888), stream=stream0)
        del primals_45
        buf56 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_14], Original ATen: [aten.clone, aten.view]
        triton_poi_fused_clone_view_17.run(buf45, buf55, buf56, 12992, 28, grid=grid(12992, 28), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_46, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf57, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf58 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf57, buf58, 464, 196, grid=grid(464, 196), stream=stream0)
        buf59 = reinterpret_tensor(buf57, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf57  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf58, primals_216, primals_217, primals_47, primals_48, buf59, 90944, grid=grid(90944), stream=stream0)
        del primals_48
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_2], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf61 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        buf72 = reinterpret_tensor(buf55, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf55  # reuse
        buf62 = reinterpret_tensor(buf72, (4, 116, 14, 14), (45472, 196, 14, 1), 0)  # alias
        buf221 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_2, getattr_l__mod___stage3___0___branch1_3, getattr_l__mod___stage3___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_20.run(buf60, primals_219, primals_220, primals_50, primals_51, buf61, buf62, buf221, 464, 196, grid=grid(464, 196), stream=stream0)
        del primals_51
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf56, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 116, 28, 28), (90944, 784, 28, 1))
        buf64 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf63, buf64, 464, 784, grid=grid(464, 784), stream=stream0)
        buf65 = reinterpret_tensor(buf63, (4, 116, 28, 28), (90944, 1, 3248, 116), 0); del buf63  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf64, primals_222, primals_223, primals_53, primals_54, buf65, 363776, grid=grid(363776), stream=stream0)
        del primals_54
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_3], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_55, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf66, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf67 = reinterpret_tensor(buf60, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf60  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf66, buf67, 464, 196, grid=grid(464, 196), stream=stream0)
        buf68 = reinterpret_tensor(buf66, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf66  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf67, primals_225, primals_226, primals_56, primals_57, buf68, 90944, grid=grid(90944), stream=stream0)
        del primals_57
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_5], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf70 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        buf71 = reinterpret_tensor(buf72, (4, 116, 14, 14), (45472, 196, 14, 1), 22736)  # alias
        buf220 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_5, getattr_l__mod___stage3___0___branch2_6, getattr_l__mod___stage3___0___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_20.run(buf69, primals_228, primals_229, primals_59, primals_60, buf70, buf71, buf220, 464, 196, grid=grid(464, 196), stream=stream0)
        del primals_60
        buf73 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf72, buf73, 181888, grid=grid(181888), stream=stream0)
        del buf62
        del buf71
        buf74 = reinterpret_tensor(buf69, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf69  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf73, buf74, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf76 = buf74; del buf74  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf75, buf76, 464, 196, grid=grid(464, 196), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf75  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf76, primals_231, primals_232, primals_62, primals_63, buf77, 90944, grid=grid(90944), stream=stream0)
        del primals_63
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_3], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf78, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf79 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf78, buf79, 464, 196, grid=grid(464, 196), stream=stream0)
        buf80 = reinterpret_tensor(buf78, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf78  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf79, primals_234, primals_235, primals_65, primals_66, buf80, 90944, grid=grid(90944), stream=stream0)
        del primals_66
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_5], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf82 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf81, buf82, 464, 196, grid=grid(464, 196), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf81  # reuse
        buf219 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_6, getattr_l__mod___stage3___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf82, primals_237, primals_238, primals_68, primals_69, buf83, buf219, 90944, grid=grid(90944), stream=stream0)
        del primals_69
        buf84 = reinterpret_tensor(buf72, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf72  # reuse
        # Source Nodes: [x_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf73, buf83, buf84, 928, 196, grid=grid(928, 196), stream=stream0)
        buf85 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf84, buf85, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf87 = buf85; del buf85  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf86, buf87, 464, 196, grid=grid(464, 196), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf86  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_1, getattr_l__mod___stage3___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf87, primals_240, primals_241, primals_71, primals_72, buf88, 90944, grid=grid(90944), stream=stream0)
        del primals_72
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_3], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf89, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf90 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf89, buf90, 464, 196, grid=grid(464, 196), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf89  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf90, primals_243, primals_244, primals_74, primals_75, buf91, 90944, grid=grid(90944), stream=stream0)
        del primals_75
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_5], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf93 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf92, buf93, 464, 196, grid=grid(464, 196), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf92  # reuse
        buf218 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_6, getattr_l__mod___stage3___2___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf93, primals_246, primals_247, primals_77, primals_78, buf94, buf218, 90944, grid=grid(90944), stream=stream0)
        del primals_78
        buf95 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf84, buf94, buf95, 928, 196, grid=grid(928, 196), stream=stream0)
        buf96 = buf94; del buf94  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf95, buf96, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf98 = buf96; del buf96  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf97, buf98, 464, 196, grid=grid(464, 196), stream=stream0)
        buf99 = reinterpret_tensor(buf97, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf97  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_1, getattr_l__mod___stage3___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf98, primals_249, primals_250, primals_80, primals_81, buf99, 90944, grid=grid(90944), stream=stream0)
        del primals_81
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_3], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf100, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf101 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf100, buf101, 464, 196, grid=grid(464, 196), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf100  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf101, primals_252, primals_253, primals_83, primals_84, buf102, 90944, grid=grid(90944), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_5], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf104 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf103, buf104, 464, 196, grid=grid(464, 196), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf103  # reuse
        buf217 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_6, getattr_l__mod___stage3___3___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf104, primals_255, primals_256, primals_86, primals_87, buf105, buf217, 90944, grid=grid(90944), stream=stream0)
        del primals_87
        buf106 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf95, buf105, buf106, 928, 196, grid=grid(928, 196), stream=stream0)
        buf107 = buf105; del buf105  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf106, buf107, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf109 = buf107; del buf107  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf108, buf109, 464, 196, grid=grid(464, 196), stream=stream0)
        buf110 = reinterpret_tensor(buf108, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf108  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_1, getattr_l__mod___stage3___4___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf109, primals_258, primals_259, primals_89, primals_90, buf110, 90944, grid=grid(90944), stream=stream0)
        del primals_90
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_3], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf111, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf112 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf111, buf112, 464, 196, grid=grid(464, 196), stream=stream0)
        buf113 = reinterpret_tensor(buf111, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf111  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf112, primals_261, primals_262, primals_92, primals_93, buf113, 90944, grid=grid(90944), stream=stream0)
        del primals_93
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_5], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf115 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf114, buf115, 464, 196, grid=grid(464, 196), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf114  # reuse
        buf216 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_6, getattr_l__mod___stage3___4___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf115, primals_264, primals_265, primals_95, primals_96, buf116, buf216, 90944, grid=grid(90944), stream=stream0)
        del primals_96
        buf117 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf106, buf116, buf117, 928, 196, grid=grid(928, 196), stream=stream0)
        buf118 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf117, buf118, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf120 = buf118; del buf118  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf119, buf120, 464, 196, grid=grid(464, 196), stream=stream0)
        buf121 = reinterpret_tensor(buf119, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf119  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_1, getattr_l__mod___stage3___5___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf120, primals_267, primals_268, primals_98, primals_99, buf121, 90944, grid=grid(90944), stream=stream0)
        del primals_99
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_3], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf122, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf123 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf122, buf123, 464, 196, grid=grid(464, 196), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf122  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf123, primals_270, primals_271, primals_101, primals_102, buf124, 90944, grid=grid(90944), stream=stream0)
        del primals_102
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_5], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf126 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf125, buf126, 464, 196, grid=grid(464, 196), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf125  # reuse
        buf215 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_6, getattr_l__mod___stage3___5___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf126, primals_273, primals_274, primals_104, primals_105, buf127, buf215, 90944, grid=grid(90944), stream=stream0)
        del primals_105
        buf128 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf117, buf127, buf128, 928, 196, grid=grid(928, 196), stream=stream0)
        buf129 = buf127; del buf127  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf128, buf129, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf131 = buf129; del buf129  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf130, buf131, 464, 196, grid=grid(464, 196), stream=stream0)
        buf132 = reinterpret_tensor(buf130, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf130  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_1, getattr_l__mod___stage3___6___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf131, primals_276, primals_277, primals_107, primals_108, buf132, 90944, grid=grid(90944), stream=stream0)
        del primals_108
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_3], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf133, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf134 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf133, buf134, 464, 196, grid=grid(464, 196), stream=stream0)
        buf135 = reinterpret_tensor(buf133, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf133  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf134, primals_279, primals_280, primals_110, primals_111, buf135, 90944, grid=grid(90944), stream=stream0)
        del primals_111
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_5], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf137 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf136, buf137, 464, 196, grid=grid(464, 196), stream=stream0)
        buf138 = reinterpret_tensor(buf136, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf136  # reuse
        buf214 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_6, getattr_l__mod___stage3___6___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf137, primals_282, primals_283, primals_113, primals_114, buf138, buf214, 90944, grid=grid(90944), stream=stream0)
        del primals_114
        buf139 = empty((4, 116, 2, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf128, buf138, buf139, 928, 196, grid=grid(928, 196), stream=stream0)
        buf140 = buf138; del buf138  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf139, buf140, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf142 = buf140; del buf140  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf141, buf142, 464, 196, grid=grid(464, 196), stream=stream0)
        buf143 = reinterpret_tensor(buf141, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf141  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_1, getattr_l__mod___stage3___7___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf142, primals_285, primals_286, primals_116, primals_117, buf143, 90944, grid=grid(90944), stream=stream0)
        del primals_117
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_3], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf144, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf145 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf144, buf145, 464, 196, grid=grid(464, 196), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf144  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf145, primals_288, primals_289, primals_119, primals_120, buf146, 90944, grid=grid(90944), stream=stream0)
        del primals_120
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_5], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 116, 14, 14), (22736, 196, 14, 1))
        buf148 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf147, buf148, 464, 196, grid=grid(464, 196), stream=stream0)
        buf149 = reinterpret_tensor(buf147, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf147  # reuse
        buf213 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_6, getattr_l__mod___stage3___7___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf148, primals_291, primals_292, primals_122, primals_123, buf149, buf213, 90944, grid=grid(90944), stream=stream0)
        del primals_123
        buf150 = empty_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37, x_39], Original ATen: [aten.clone, aten.view]
        triton_poi_fused_clone_view_28.run(buf139, buf149, buf150, 12992, 14, grid=grid(12992, 14), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf151, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf152 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf151, buf152, 928, 49, grid=grid(928, 49), stream=stream0)
        buf153 = reinterpret_tensor(buf151, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf151  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf152, primals_294, primals_295, primals_125, primals_126, buf153, 45472, grid=grid(45472), stream=stream0)
        del primals_126
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_2], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf155 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf149, (4, 464, 7, 7), (22736, 49, 7, 1), 0); del buf149  # reuse
        buf156 = reinterpret_tensor(buf166, (4, 232, 7, 7), (22736, 49, 7, 1), 0)  # alias
        buf212 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_2, getattr_l__mod___stage4___0___branch1_3, getattr_l__mod___stage4___0___branch1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_31.run(buf154, primals_297, primals_298, primals_128, primals_129, buf155, buf156, buf212, 928, 49, grid=grid(928, 49), stream=stream0)
        del primals_129
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf150, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 232, 14, 14), (45472, 196, 14, 1))
        buf158 = empty_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf157, buf158, 928, 196, grid=grid(928, 196), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (4, 232, 14, 14), (45472, 1, 3248, 232), 0); del buf157  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_33.run(buf158, primals_300, primals_301, primals_131, primals_132, buf159, 181888, grid=grid(181888), stream=stream0)
        del primals_132
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_3], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf160, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf161 = reinterpret_tensor(buf154, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf154  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf160, buf161, 928, 49, grid=grid(928, 49), stream=stream0)
        buf162 = reinterpret_tensor(buf160, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf161, primals_303, primals_304, primals_134, primals_135, buf162, 45472, grid=grid(45472), stream=stream0)
        del primals_135
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_5], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf164 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        buf165 = reinterpret_tensor(buf166, (4, 232, 7, 7), (22736, 49, 7, 1), 11368)  # alias
        buf211 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_5, getattr_l__mod___stage4___0___branch2_6, getattr_l__mod___stage4___0___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_34.run(buf163, primals_306, primals_307, primals_137, primals_138, buf164, buf165, buf211, 928, 49, grid=grid(928, 49), stream=stream0)
        del primals_138
        buf167 = empty((4, 232, 2, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf166, buf167, 90944, grid=grid(90944), stream=stream0)
        del buf156
        del buf165
        buf168 = reinterpret_tensor(buf163, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf163  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf167, buf168, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf170 = buf168; del buf168  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf169, buf170, 928, 49, grid=grid(928, 49), stream=stream0)
        buf171 = reinterpret_tensor(buf169, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf169  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf170, primals_309, primals_310, primals_140, primals_141, buf171, 45472, grid=grid(45472), stream=stream0)
        del primals_141
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_3], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf172, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf173 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf172, buf173, 928, 49, grid=grid(928, 49), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf172  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf173, primals_312, primals_313, primals_143, primals_144, buf174, 45472, grid=grid(45472), stream=stream0)
        del primals_144
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_5], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf176 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf175, buf176, 928, 49, grid=grid(928, 49), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf175  # reuse
        buf210 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_6, getattr_l__mod___stage4___1___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38.run(buf176, primals_315, primals_316, primals_146, primals_147, buf177, buf210, 45472, grid=grid(45472), stream=stream0)
        del primals_147
        buf178 = reinterpret_tensor(buf166, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf166  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf167, buf177, buf178, 1856, 49, grid=grid(1856, 49), stream=stream0)
        buf179 = buf177; del buf177  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf178, buf179, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf181 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf180, buf181, 928, 49, grid=grid(928, 49), stream=stream0)
        buf182 = reinterpret_tensor(buf180, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf180  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_1, getattr_l__mod___stage4___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf181, primals_318, primals_319, primals_149, primals_150, buf182, 45472, grid=grid(45472), stream=stream0)
        del primals_150
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_3], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf183, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf184 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf183, buf184, 928, 49, grid=grid(928, 49), stream=stream0)
        buf185 = reinterpret_tensor(buf183, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf183  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf184, primals_321, primals_322, primals_152, primals_153, buf185, 45472, grid=grid(45472), stream=stream0)
        del primals_153
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_5], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf187 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf186, buf187, 928, 49, grid=grid(928, 49), stream=stream0)
        buf188 = reinterpret_tensor(buf186, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf186  # reuse
        buf209 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_6, getattr_l__mod___stage4___2___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38.run(buf187, primals_324, primals_325, primals_155, primals_156, buf188, buf209, 45472, grid=grid(45472), stream=stream0)
        del primals_156
        buf189 = empty((4, 232, 2, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf178, buf188, buf189, 1856, 49, grid=grid(1856, 49), stream=stream0)
        buf190 = buf188; del buf188  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf189, buf190, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf192 = buf190; del buf190  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf191, buf192, 928, 49, grid=grid(928, 49), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf191  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_1, getattr_l__mod___stage4___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf192, primals_327, primals_328, primals_158, primals_159, buf193, 45472, grid=grid(45472), stream=stream0)
        del primals_159
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_3], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf194, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf195 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf194, buf195, 928, 49, grid=grid(928, 49), stream=stream0)
        buf196 = reinterpret_tensor(buf194, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf194  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf195, primals_330, primals_331, primals_161, primals_162, buf196, 45472, grid=grid(45472), stream=stream0)
        del primals_162
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_5], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 232, 7, 7), (11368, 49, 7, 1))
        buf198 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf197, buf198, 928, 49, grid=grid(928, 49), stream=stream0)
        buf199 = reinterpret_tensor(buf197, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf197  # reuse
        buf208 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_6, getattr_l__mod___stage4___3___branch2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38.run(buf198, primals_333, primals_334, primals_164, primals_165, buf199, buf208, 45472, grid=grid(45472), stream=stream0)
        del primals_165
        buf200 = empty_strided((4, 464, 7, 7), (22736, 1, 3248, 464), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50, x_52], Original ATen: [aten.clone, aten.view]
        triton_poi_fused_clone_view_40.run(buf189, buf199, buf200, 12992, 7, grid=grid(12992, 7), stream=stream0)
        del buf199
        # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf202 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf201, buf202, 4096, 49, grid=grid(4096, 49), stream=stream0)
        buf203 = reinterpret_tensor(buf201, (4, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf201  # reuse
        buf207 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___conv5_1, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_42.run(buf202, primals_336, primals_337, primals_167, primals_168, buf203, buf207, 200704, grid=grid(200704), stream=stream0)
        del primals_168
        buf204 = empty((4, 1024), device='cuda', dtype=torch.float32)
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.mean]
        triton_per_fused_mean_43.run(buf205, buf203, 4096, 49, grid=grid(4096), stream=stream0)
        del buf203
        buf206 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_170, buf205, reinterpret_tensor(primals_169, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf206)
        del primals_170
        return (buf206, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, buf1, buf3, buf4, buf5, buf6, buf8, buf9, buf11, buf14, buf15, buf17, buf18, buf20, reinterpret_tensor(buf23, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf26, buf27, buf29, buf30, buf32, reinterpret_tensor(buf34, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf37, buf38, buf40, buf41, buf43, reinterpret_tensor(buf45, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf48, buf49, buf51, buf52, buf54, buf56, buf58, buf59, buf61, buf64, buf65, buf67, buf68, buf70, reinterpret_tensor(buf73, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf76, buf77, buf79, buf80, buf82, reinterpret_tensor(buf84, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf87, buf88, buf90, buf91, buf93, reinterpret_tensor(buf95, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf98, buf99, buf101, buf102, buf104, reinterpret_tensor(buf106, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf109, buf110, buf112, buf113, buf115, reinterpret_tensor(buf117, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf120, buf121, buf123, buf124, buf126, reinterpret_tensor(buf128, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf131, buf132, buf134, buf135, buf137, reinterpret_tensor(buf139, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf142, buf143, buf145, buf146, buf148, buf150, buf152, buf153, buf155, buf158, buf159, buf161, buf162, buf164, reinterpret_tensor(buf167, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf170, buf171, buf173, buf174, buf176, reinterpret_tensor(buf178, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf181, buf182, buf184, buf185, buf187, reinterpret_tensor(buf189, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf192, buf193, buf195, buf196, buf198, buf200, buf202, buf205, reinterpret_tensor(primals_169, (1000, 1024), (1024, 1), 0), buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf225, buf226, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
