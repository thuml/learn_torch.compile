
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


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fjmeqaknlpk5m5fqj6h6x72sutinrgbi6evb72wghecwgcpgd3.py
# Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 225)
    x2 = xindex % 225
    y4 = yindex
    x5 = xindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 224, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (224*x3) + (50176*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (y0 + (3*x5) + (151875*y1)), tmp8, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2s/c2su6u62tlywk2wwcmiaqmeerposmvshczsj5okff2heff2im3si.py
# Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
# x => constant_pad_nd
# x_1 => convolution
triton_poi_fused_constant_pad_nd_convolution_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fugtcurprmfqcmmsv4wb65lm5b62eairtx6o5fseokmsz6kljj.py
# Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_2 => add_1, mul_1, mul_2, sub
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3icmvcsuowveo3xi3q6knykx5hd5xgrgawrvo6s3g2bdrdupgd6.py
# Source Nodes: [shortcut_1, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_1 => add_6
# x_13 => add_5, mul_7, mul_8, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlnzgbwuhpp6rionij4wdjnrpdlwffznpjjw5yiooun5udvnpvg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 => convolution_3
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camoqwkujz3yobsyf5dgfi526dcmuqnkg5hjdzhzjvproqwssxmm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 => convolution_4
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (200704 + x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldodenumt4pdt527enal5yzvbzb36tadwgamv7axq5u2b4q7nm5.py
# Source Nodes: [cat_81, x_20, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_81 => cat
# x_20 => add_8, mul_10, mul_11, sub_3
# x_23 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 12544) % 192
    x2 = (xindex // 2408448)
    x3 = xindex % 2408448
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (1204224*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 192, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-1204224) + x3 + (1204224*x2)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzrgbh362r55sfwn7nizumflcx2imwhaswducbcq36mzrrnh5hw.py
# Source Nodes: [x_25], Original ATen: [aten.constant_pad_nd]
# x_25 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12769
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 113)
    x2 = xindex % 113
    y0 = yindex % 64
    y1 = (yindex // 64)
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1, 1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (y0 + (64*x4) + (817216*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmagxocnriti3fezwkmqtn3rqjh6y4f72wcimhttakmqsqrk4ch.py
# Source Nodes: [x_27], Original ATen: [aten.constant_pad_nd]
# x_27 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 13225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 115)
    x2 = xindex % 115
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (802703 + x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (y0 + (64*x5) + (846400*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhhevviyma4kjsrvophymhovqxpcksyngcvb2gmz7ogmeqgeqkf.py
# Source Nodes: [x_29], Original ATen: [aten.constant_pad_nd]
# x_29 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 13689
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 117)
    x2 = xindex % 117
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (1605406 + x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (y0 + (64*x5) + (876096*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogqsx52r7auhu3wmolmsy6u7rky3znd3fcvvm2nmyyeqfex7um7.py
# Source Nodes: [cat_80, x_32, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_80 => cat_1
# x_32 => add_10, mul_13, mul_14, sub_4
# x_35 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 192
    x2 = (xindex // 602112)
    x3 = xindex % 602112
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (200704*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-200704) + x3 + (200704*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-401408) + x3 + (200704*x2)), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tl.store(out_ptr0 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvoyos46jdhosvespqjvv765byttn2miwgqzi3o22qaabe2ccfom.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0 => convolution_8
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgu2nqsinhfimgzrw54m5ehxgi5x77sahhtn3i3c7v27me5cvrel.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1 => convolution_9
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (301056 + x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxenkq2rrb76qqdh4exrez5hj6aosxm7pvii22ag7gc6ihowfprp.py
# Source Nodes: [cat_79, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat]
# cat_79 => cat_2
# x_39 => add_12, mul_16, mul_17, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_cat_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 40
    x2 = (xindex // 125440)
    x3 = xindex % 125440
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (62720*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-62720) + x3 + (62720*x2)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr0 + (x4), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/csh3zpfsqclftgng7ccs75tbqqmr6xf4zkil6x2w3n74d6ucswb6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____1___conv_pw_0 => convolution_10
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 20
    y1 = (yindex // 20)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (20*x2) + (62720*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgx4ozoby4r6fux6ety7lx42yxamwt4kt3ockbfyisvn46smvqvl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____1___conv_pw_1 => convolution_11
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 20
    y1 = (yindex // 20)
    tmp0 = tl.load(in_ptr0 + (62720 + x2 + (3136*y0) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (20*x2) + (62720*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijknzkwtmbt5oa374fc6ltjjicfo7pxypm4va3wqmkjcuaxjqhz.py
# Source Nodes: [cat_78, x_45, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_78 => cat_3
# x_45 => add_14, mul_19, mul_20, sub_6
# x_48 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 120
    x2 = xindex
    y1 = (yindex // 120)
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (188160*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-188160) + x2 + (3136*y0) + (188160*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (120*x2) + (376320*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2osu2fcribuyjcspg7hmcvgsgw6gqwmjz26cblgpqgzq4cg3ayv.py
# Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_50 => add_16, mul_22, mul_23, sub_7
# x_53 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 120
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/uc/cucphoz3ydzc2yeerfg7iblxlp2bxr55lu4jlfiva2lhgll4ycym.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0 => convolution_13
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (376320*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camnsctm5ll65rnh2jwbhdzj6nrrsfsurymnltfqpegahmac2icd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1 => convolution_14
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (188160 + x2 + (3136*y0) + (376320*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckuqsw5snbjukzzpkvznlrg75c4nyiqav26cuq6hnlzga67bk6kj.py
# Source Nodes: [cat_77, shortcut_3, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_77 => cat_4
# shortcut_3 => add_19
# x_57 => add_18, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 40
    x2 = xindex
    y1 = (yindex // 40)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (62720*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-62720) + x2 + (3136*y0) + (62720*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (y0 + (40*x2) + (125440*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f7/cf7czcblh7lzxsmk6we5gosmzyxmrnmimixv4ehloekqf7qecxei.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_63 => add_21, mul_28, mul_29, sub_9
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 240
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2vsfkos27swkzlfcfbdgcehvxp5zocsjrjn4mlix45thxd6nvh.py
# Source Nodes: [x_68], Original ATen: [aten.constant_pad_nd]
# x_68 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3249
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 57)
    x2 = xindex % 57
    y0 = yindex % 60
    y1 = (yindex // 60)
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1, 1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (56*x3) + (3136*y0) + (752640*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (y0 + (60*x4) + (194940*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3ksyup6eb23ijf2jpe75v3d57dgyc2lxeiqr5pktm3bu2z6m7uk.py
# Source Nodes: [x_70], Original ATen: [aten.constant_pad_nd]
# x_70 => constant_pad_nd_5
triton_poi_fused_constant_pad_nd_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3481
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 59)
    x2 = xindex % 59
    y0 = yindex % 60
    y1 = (yindex // 60)
    x5 = xindex
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (188103 + x2 + (56*x3) + (3136*y0) + (752640*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (60*x5) + (208860*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3joflmlv57eihiym6giy2stg6rujwsyegqqocy5xofp66grklt6.py
# Source Nodes: [x_72], Original ATen: [aten.constant_pad_nd]
# x_72 => constant_pad_nd_6
triton_poi_fused_constant_pad_nd_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3721
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 61)
    x2 = xindex % 61
    y0 = yindex % 60
    y1 = (yindex // 60)
    x5 = xindex
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (376206 + x2 + (56*x3) + (3136*y0) + (752640*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (60*x5) + (223260*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndp57dxinja62uymfvijzm2hict6zqkhqnydz3a3qxuxarjuwgr.py
# Source Nodes: [x_74], Original ATen: [aten.constant_pad_nd]
# x_74 => constant_pad_nd_7
triton_poi_fused_constant_pad_nd_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3969
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 63)
    x2 = xindex % 63
    y0 = yindex % 60
    y1 = (yindex // 60)
    x5 = xindex
    tmp0 = (-3) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (564309 + x2 + (56*x3) + (3136*y0) + (752640*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (60*x5) + (238140*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqjum3tgwyt5y6676tsmxkvgz27z7chtr2zw75zfapmubsazxfq7.py
# Source Nodes: [cat_76, x_77, x_80, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_76 => cat_5
# x_77 => add_23, mul_32, mul_33, sub_10
# x_80 => mul_34, sigmoid_1
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel):
    xnumel = 1920
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 240
    r2 = rindex
    x1 = (xindex // 240)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (784*x0) + (47040*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-47040) + r2 + (784*x0) + (47040*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 180, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-94080) + r2 + (784*x0) + (47040*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 240, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-141120) + r2 + (784*x0) + (47040*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = triton_helpers.promote_to_tensor(tl.sum(tmp49, 0))
    tmp51 = 784.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (784*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnseug323o3mzpttnm5cugg6j4on24wl36axgvzwogr2zezypk5.py
# Source Nodes: [x_80, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_80 => mul_34, sigmoid_1
# x_se => mean
# x_se_1 => convolution_20
# x_se_2 => mul_35, sigmoid_2
triton_poi_fused_convolution_mean_silu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacnycexnkh2kymc6zs2ah2rk4l7o4iafzoydbia6xftc7ues4nk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
# x_80 => mul_34, sigmoid_1
# x_81 => mul_36
# x_se => mean
# x_se_1 => convolution_20
# x_se_2 => mul_35, sigmoid_2
# x_se_3 => convolution_21
triton_poi_fused_convolution_mean_mul_sigmoid_silu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwdltobjk5zde45bpvyhdnviuzha36hayfnyunwb4cn2hi2dsho.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_83 => add_25, mul_38, mul_39, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciau3lc2r5hrlaq64arxgih63xcynzx7gpjvljeeb4depean4f5m.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_pw_0 => convolution_23
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykhggktwhhrnz7ic3ijiz2mbdwf5li6pxobjvfr26a3u4n4pvuc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_pw_1 => convolution_24
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (21952 + x2 + (784*y0) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcgdfkgyj6pwio4uaqhmbe5c53ig2pz2lw3ga4mkvijlf67nrrp.py
# Source Nodes: [cat_75, x_89, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
# cat_75 => cat_6
# x_89 => add_27, mul_41, mul_42, sub_12
# x_92 => mul_43, sigmoid_4
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 336
    x2 = (xindex // 263424)
    x3 = xindex % 263424
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + x3 + (131712*x2)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkm2migkmgb2fgeyfjjlce2pvkulzwu2qcmix6vp73nkrrsbl2z.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 => convolution_25
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqwf74hyj4f3xtjekd5ttsqs2tyn5xraoslm7b7z5uitjsx47qa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 => convolution_26
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chrdm24wbp4jomgw5zjgiognbblityvhtvkz2w6qoxxek6zyb3ov.py
# Source Nodes: [cat_74, x_95, x_98, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_74 => cat_7
# x_95 => add_29, mul_45, mul_46, sub_13
# x_98 => mul_47, sigmoid_5
# x_se_4 => mean_1
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 2688
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 336
    r2 = rindex
    x1 = (xindex // 336)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (784*x0) + (131712*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + r2 + (784*x0) + (131712*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 784.0
    tmp36 = tmp34 / tmp35
    tl.store(out_ptr0 + (r2 + (784*x3)), tmp28, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjknn3lagmfo43yzalw3qy4acbjymqui4jqnyk32fvzdiclglza5.py
# Source Nodes: [x_98, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_98 => mul_47, sigmoid_5
# x_se_4 => mean_1
# x_se_5 => convolution_27
# x_se_6 => mul_48, sigmoid_6
triton_poi_fused_convolution_mean_silu_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwox3au4gonczbhmuusiz4tm2fzlh2x44jnkj5a7ylvcv5yngs7d.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_98, x_99, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_7
# x_98 => mul_47, sigmoid_5
# x_99 => mul_49
# x_se_4 => mean_1
# x_se_5 => convolution_27
# x_se_6 => mul_48, sigmoid_6
# x_se_7 => convolution_28
triton_poi_fused_convolution_mean_mul_sigmoid_silu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 336
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7qziq3xp3aqbigfmgk5n3ohn5rbr7tbyzgcooxbhgguwtalvv4.py
# Source Nodes: [cat_73, shortcut_5, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_73 => cat_8
# shortcut_5 => add_32
# x_102 => add_31, mul_51, mul_52, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 56
    x2 = (xindex // 43904)
    x3 = xindex % 43904
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (21952*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 56, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-21952) + x3 + (21952*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cih7uobxh7436ryy6xpzdv7ajdovoc3dx2xp6sg34r6hwr52in6l.py
# Source Nodes: [cat_67, shortcut_7, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_67 => cat_14
# shortcut_7 => add_46
# x_142 => add_45, mul_77, mul_78, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 56
    x2 = xindex
    y1 = (yindex // 56)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (21952*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 56, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-21952) + x2 + (784*y0) + (21952*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrx6zwxlfuyiuiynieuksbyzfn76nscafa4lv3pagb3fegjchfw.py
# Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_148 => add_48, mul_80, mul_81, sub_21
triton_poi_fused__native_batch_norm_legit_no_training_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 336
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catk52xyeoc3spxdndchl3bpkiw6hbz53j4gwpje6wvtstcb4jfy.py
# Source Nodes: [x_153], Original ATen: [aten.constant_pad_nd]
# x_153 => constant_pad_nd_8
triton_poi_fused_constant_pad_nd_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 841
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 29)
    x2 = xindex % 29
    y0 = yindex % 112
    y1 = (yindex // 112)
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1, 1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (28*x3) + (784*y0) + (263424*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (y0 + (112*x4) + (94192*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijcqmyhqhwxlthf7dhoep6qje27mfhw54yuhimd7m3xhu37vni4.py
# Source Nodes: [x_155], Original ATen: [aten.constant_pad_nd]
# x_155 => constant_pad_nd_9
triton_poi_fused_constant_pad_nd_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 31)
    x2 = xindex % 31
    y0 = yindex % 112
    y1 = (yindex // 112)
    x5 = xindex
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (87779 + x2 + (28*x3) + (784*y0) + (263424*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (112*x5) + (107632*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnzsgzvteouwohafp6qepdeom6r5dzqf6pzn5snxtu3unwsbnlt.py
# Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
# x_157 => constant_pad_nd_10
triton_poi_fused_constant_pad_nd_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 1089
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 33)
    x2 = xindex % 33
    y0 = yindex % 112
    y1 = (yindex // 112)
    x5 = xindex
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (175558 + x2 + (28*x3) + (784*y0) + (263424*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (112*x5) + (121968*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilxuadd7uokb2ox6o3pgftf4azpvzi5jy7nblaik4nivsm4ouq2.py
# Source Nodes: [cat_66, x_160, x_163, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_66 => cat_15
# x_160 => add_50, mul_84, mul_85, sub_22
# x_163 => mul_86, sigmoid_17
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 336
    r2 = rindex
    x1 = (xindex // 336)
    x3 = xindex
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (21952*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 224, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-21952) + r2 + (196*x0) + (21952*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 336, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-43904) + r2 + (196*x0) + (21952*x1)), rmask & tmp15 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sigmoid(tmp36)
    tmp38 = tmp36 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None]
    tmp43 = 196.0
    tmp44 = tmp42 / tmp43
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp36, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7ydaokntbyiuflxlqmlrqbjtwdv64sfqcpm6523q7niakc32it.py
# Source Nodes: [x_163, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_163 => mul_86, sigmoid_17
# x_se_16 => mean_4
# x_se_17 => convolution_51
# x_se_18 => mul_87, sigmoid_18
triton_poi_fused_convolution_mean_silu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgr34dkccipj5lclza4ppconpftdot5tc64773uhut53zgbzex6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
# x_163 => mul_86, sigmoid_17
# x_164 => mul_88
# x_se_16 => mean_4
# x_se_17 => convolution_51
# x_se_18 => mul_87, sigmoid_18
# x_se_19 => convolution_52
triton_poi_fused_convolution_mean_mul_sigmoid_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cex4r4f4njjwoiavpilmt5uonhcwyt2f6kmilhfhpgvspfhiblcx.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_166 => add_52, mul_90, mul_91, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 104
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfvnxqbnds5hiipksng2jevipzqmqqonmd5wdz2vl4fvtq45vmo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pw_0 => convolution_54
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (10192*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyyowjjwkqlkx4sdvfttes744tcbuv3svn6vnscxfv5hgo65oy2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pw_1 => convolution_55
triton_poi_fused_convolution_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (10192 + x2 + (196*y0) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (10192*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxc2c7ujuliys3htlwncqaemssbe66ejdic365j6bwvd5t2a272.py
# Source Nodes: [cat_65, x_172, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
# cat_65 => cat_16
# x_172 => add_54, mul_93, mul_94, sub_24
# x_175 => mul_95, sigmoid_20
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 624
    x2 = (xindex // 122304)
    x3 = xindex % 122304
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (61152*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 624, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-61152) + x3 + (61152*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr5jdrmk23ieyijwihn6225ativz6awdkx6njyr5vig2cma5hlp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 => convolution_56
triton_poi_fused_convolution_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlifgomckn7wmv45jcckvafi3rrwfkzx35w6mynauspdlfodt7y.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 => convolution_57
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (30576 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnq2yuz3pjrpafqieibcryvono7awrdyjoqvo6q2qlntfe6cfht.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 => convolution_58
triton_poi_fused_convolution_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdco7kh6oraxygkyu6rl4dvsix2wahthinz3224c6iu5f46aw4b.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 => convolution_59
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (91728 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citzzco2ttix7bxts2a3b6qa2trk2hy5ds6ylylxehs7i2gisaec.py
# Source Nodes: [cat_64, x_178, x_181, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_64 => cat_17
# x_178 => add_56, mul_97, mul_98, sub_25
# x_181 => mul_99, sigmoid_21
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 624
    r2 = rindex
    x1 = (xindex // 624)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 156, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (30576*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 312, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-30576) + r2 + (196*x0) + (30576*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 468, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-61152) + r2 + (196*x0) + (30576*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 624, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-91728) + r2 + (196*x0) + (30576*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 196.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/csztn7g5ho264wnjtwgxceitkywowmgznuqlxqp6glcthgc3exgg.py
# Source Nodes: [x_181, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_181 => mul_99, sigmoid_21
# x_se_20 => mean_5
# x_se_21 => convolution_60
# x_se_22 => mul_100, sigmoid_22
triton_poi_fused_convolution_mean_silu_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 26
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/ciguczk7qoikdnvy725ukdneagej7h74vruphot4rwzqckxqtfpa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_181, x_182, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_23
# x_181 => mul_99, sigmoid_21
# x_182 => mul_101
# x_se_20 => mean_5
# x_se_21 => convolution_60
# x_se_22 => mul_100, sigmoid_22
# x_se_23 => convolution_61
triton_poi_fused_convolution_mean_mul_sigmoid_silu_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 624
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx25izfgitcur2n652iufpi45d5uazguvha3bxv7gioijesx67yk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 => convolution_62
triton_poi_fused_convolution_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpeixdqs3xx7qquykkgovkmplvtduts6hsf2e34ud7k2hhcmxzde.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 => convolution_63
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
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2v2jkaaoejk2bj7ce4ohz2x2zsmivjtefycekikh2clsu2xrt3.py
# Source Nodes: [cat_63, shortcut_9, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_63 => cat_18
# shortcut_9 => add_59
# x_185 => add_58, mul_103, mul_104, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 104
    x2 = (xindex // 20384)
    x3 = xindex % 20384
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (10192*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 104, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10192) + x3 + (10192*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdphfuhvpvvozo5aol325bcxd5g2xuryljddaovsakpqycwy6t7.py
# Source Nodes: [cat_57, shortcut_11, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_57 => cat_24
# shortcut_11 => add_73
# x_225 => add_72, mul_129, mul_130, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 104
    x2 = xindex
    y1 = (yindex // 104)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (10192*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 104, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10192) + x2 + (196*y0) + (10192*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (y0 + (104*x2) + (20384*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvafvy7coiyjqimoiukfwqbczyspo3kbngugburiyu4flrgled3p.py
# Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_231 => add_75, mul_132, mul_133, sub_33
# x_234 => mul_134, sigmoid_32
triton_poi_fused__native_batch_norm_legit_no_training_silu_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (624*x2) + (122304*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncqn5gjw6lfp7uiqg7lyp3v4imihn5rpaog7ew2l55qkif7oyt5.py
# Source Nodes: [x_236, x_239, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_236 => add_77, mul_136, mul_137, sub_34
# x_239 => mul_138, sigmoid_33
# x_se_32 => mean_8
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_63', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 624
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnsvynhj7t3nf73liyrhs5ch3p2uww67wmhsgtt2tddnnxdv4rj.py
# Source Nodes: [x_239, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_239 => mul_138, sigmoid_33
# x_se_32 => mean_8
# x_se_33 => convolution_86
# x_se_34 => mul_139, sigmoid_34
triton_poi_fused_convolution_mean_silu_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 52
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwzgvotghdgegn4zoh7ta4at6ifrdjpufbno2n5er5mibjs7e63.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_239 => mul_138, sigmoid_33
# x_240 => mul_140
# x_se_32 => mean_8
# x_se_33 => convolution_86
# x_se_34 => mul_139, sigmoid_34
# x_se_35 => convolution_87
triton_poi_fused_convolution_mean_mul_sigmoid_silu_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (624*x2) + (122304*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaybf3jtcie7aphdkoawwnnf6gfzelfyjxr7bpr24uweru7zbv4.py
# Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_242 => add_79, mul_142, mul_143, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_66', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 160
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4katp7nrmbcw27xy3r4n2dymw6c7njnrh5nugzp2oakuseagl3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pw_0 => convolution_89
triton_poi_fused_convolution_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74tidexrz33g2d2snxkzkkovty6yqcmeet63a7bmp5omxiwfb2n.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pw_1 => convolution_90
triton_poi_fused_convolution_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (15680 + x2 + (196*y0) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4st4jawukvyy7ug4mapbgkxsaunv443zxunb676l27tf2wzsq7.py
# Source Nodes: [cat_56, x_248, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
# cat_56 => cat_25
# x_248 => add_81, mul_145, mul_146, sub_36
# x_251 => mul_147, sigmoid_36
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 480
    x2 = (xindex // 94080)
    x3 = xindex % 94080
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (47040*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + x3 + (47040*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvcnd5esnx4kmikb47oyjvxcpxqt7w7bjhlrifm5ubexabhduuo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 => convolution_91
triton_poi_fused_convolution_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jecrvvsykhmnhpjumsrjd3iouxkeavwdq4uqr24gigkuou7eqo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 => convolution_92
triton_poi_fused_convolution_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (23520 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdaevb4sewffpoz6wl4i5aebluszon6jmu2ppiyydf22zhyqvsck.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 => convolution_93
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
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjvgtri5a5d5m7vpa7qvardbupdlogopnmgntymhjqmodotdhjn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 => convolution_94
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (70560 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3fwrrvmbakuv4l5p2mgwnr7uc4uu2o7dyxwmvofagkcvcu56gz.py
# Source Nodes: [cat_55, x_254, x_257, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_55 => cat_26
# x_254 => add_83, mul_149, mul_150, sub_37
# x_257 => mul_151, sigmoid_37
# x_se_36 => mean_9
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_74', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 480
    r2 = rindex
    x1 = (xindex // 480)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 120, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (23520*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 240, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-23520) + r2 + (196*x0) + (23520*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 360, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-47040) + r2 + (196*x0) + (23520*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 480, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-70560) + r2 + (196*x0) + (23520*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 196.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf34aanlrk35b2ho2jzaal2wkshfa6aia3oea3taxpl64drokzua.py
# Source Nodes: [x_257, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_257 => mul_151, sigmoid_37
# x_se_36 => mean_9
# x_se_37 => convolution_95
# x_se_38 => mul_152, sigmoid_38
triton_poi_fused_convolution_mean_silu_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chrdnx34j3q4aoesnqsjp3ac7ytrfn56g7hgoslt4xtlj35wcrf2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_257, x_258, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_257 => mul_151, sigmoid_37
# x_258 => mul_153
# x_se_36 => mean_9
# x_se_37 => convolution_95
# x_se_38 => mul_152, sigmoid_38
# x_se_39 => convolution_96
triton_poi_fused_convolution_mean_mul_sigmoid_silu_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_76', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 480
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpf7ki3a3huiezupkwm5y4g5fqx7e5bxe3hla5ubny4crqwwf3wj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 => convolution_97
triton_poi_fused_convolution_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5izell4m2aigedo4bhde5r7khjentkkfsqghrakp4o7w4qnyet.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 => convolution_98
triton_poi_fused_convolution_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqlas54oo3g3u2x7x55prqmzm4vaqylht74uq7tgd7nv4dtvfxo.py
# Source Nodes: [cat_54, shortcut_13, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_54 => cat_27
# shortcut_13 => add_86
# x_261 => add_85, mul_155, mul_156, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 160
    x2 = (xindex // 31360)
    x3 = xindex % 31360
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (15680*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-15680) + x3 + (15680*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x4), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qv3c6szcdblh54ngqbr5xncn2rpvgwn4mpsh3ma7iy4vv7dhob.py
# Source Nodes: [cat_48, shortcut_15, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_48 => cat_33
# shortcut_15 => add_100
# x_301 => add_99, mul_181, mul_182, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 160
    x2 = xindex
    y1 = (yindex // 160)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (15680*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-15680) + x2 + (196*y0) + (15680*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (y0 + (160*x2) + (31360*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplrvs225nlznsyqxxgr7gonvjedxhykdd5ucfdcl7vc2mfijquw.py
# Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_307 => add_102, mul_184, mul_185, sub_45
triton_poi_fused__native_batch_norm_legit_no_training_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 960
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5cfhugqzsuiad5ob4p7bpc2ajridev4jp6wvwb3z6x7obgnwx2s.py
# Source Nodes: [x_312], Original ATen: [aten.constant_pad_nd]
# x_312 => constant_pad_nd_11
triton_poi_fused_constant_pad_nd_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 15)
    x2 = xindex % 15
    y0 = yindex % 240
    y1 = (yindex // 240)
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1, 1], 14, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (14*x3) + (196*y0) + (188160*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (y0 + (240*x4) + (54000*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wfhsynvrnbjhqku437xhwb257orxbkdc4lps2cxw2htbrwfleg.py
# Source Nodes: [x_314], Original ATen: [aten.constant_pad_nd]
# x_314 => constant_pad_nd_12
triton_poi_fused_constant_pad_nd_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 17)
    x2 = xindex % 17
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (47025 + x2 + (14*x3) + (196*y0) + (188160*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (240*x5) + (69360*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbykqqyomzo2ymlytu6myfba2gfib4djzw57s3m5zlu5lj5ok4w.py
# Source Nodes: [x_316], Original ATen: [aten.constant_pad_nd]
# x_316 => constant_pad_nd_13
triton_poi_fused_constant_pad_nd_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 361
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 19)
    x2 = xindex % 19
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (94050 + x2 + (14*x3) + (196*y0) + (188160*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (240*x5) + (86640*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpbzrkcq2mwgm4p3nsdaqmu4slu5vn4f4cmioccloxpdwo3gbc7.py
# Source Nodes: [x_318], Original ATen: [aten.constant_pad_nd]
# x_318 => constant_pad_nd_14
triton_poi_fused_constant_pad_nd_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 21)
    x2 = xindex % 21
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    tmp0 = (-3) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (141075 + x2 + (14*x3) + (196*y0) + (188160*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (240*x5) + (105840*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b4/cb46rrljvkd7dmffmxqqyjhsbwevi3cblw54hkcvxlldjecewclx.py
# Source Nodes: [cat_47, x_321, x_324, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_47 => cat_34
# x_321 => add_104, mul_188, mul_189, sub_46
# x_324 => mul_190, sigmoid_49
# x_se_48 => mean_12
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 960
    r2 = rindex
    x1 = (xindex // 960)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (49*x0) + (11760*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-11760) + r2 + (49*x0) + (11760*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 720, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-23520) + r2 + (49*x0) + (11760*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 960, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-35280) + r2 + (49*x0) + (11760*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 49.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qbpf47nljagchkcnmdktwdygyfkfjnpzjnon2ywadv6abkm7rx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
# x_324 => mul_190, sigmoid_49
# x_325 => mul_192
# x_se_48 => mean_12
# x_se_49 => convolution_124
# x_se_50 => mul_191, sigmoid_50
# x_se_51 => convolution_125
triton_poi_fused_convolution_mean_mul_sigmoid_silu_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfguizy5aocn2jcx4wzvz4s7bmxi5brsytf2u5owoegbq6uisae.py
# Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_327 => add_106, mul_194, mul_195, sub_47
triton_poi_fused__native_batch_norm_legit_no_training_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (264*x2) + (12936*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefi4migmbalv5eyhh23qe6fyj7qcnkod5nihteujdklfyi46hol.py
# Source Nodes: [x_332, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_332 => add_108, mul_197, mul_198, sub_48
# x_335 => mul_199, sigmoid_52
triton_poi_fused__native_batch_norm_legit_no_training_silu_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_89', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 620928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1584
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35rw7g2ppmz6xamguucfrsxbgz5n4tv7wxloijluungkazubk3e.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 => convolution_128
triton_poi_fused_convolution_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coydvbzhoe6fw75tkkoqrleoqpdpdixfbxwabqayz464ejmvvelc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 => convolution_129
triton_poi_fused_convolution_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (19404 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7e3tjt3nx3tbiegtg37dh55nrmqvwyci4bypm4aa2othizsnm3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 => convolution_130
triton_poi_fused_convolution_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxjasmesekp755qoxn6r6ssiviutkwndaev5f7gz4osldvkim65.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 => convolution_131
triton_poi_fused_convolution_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (58212 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4ioylw2oofa3qomjae5cql4lkvhaobcem425zd2sakzserlxs5.py
# Source Nodes: [cat_46, x_338, x_341, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
# cat_46 => cat_35
# x_338 => add_110, mul_201, mul_202, sub_49
# x_341 => mul_203, sigmoid_53
# x_se_52 => mean_13
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_94', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12672
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 1584
    r2 = rindex
    x1 = (xindex // 1584)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 396, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (49*x0) + (19404*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 792, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-19404) + r2 + (49*x0) + (19404*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 1188, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-38808) + r2 + (49*x0) + (19404*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1584, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-58212) + r2 + (49*x0) + (19404*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 49.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinqe2n4dgjieeyf6niei6t3sznyn23b7qzfx7skn4vncocwre2z.py
# Source Nodes: [x_341, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_341 => mul_203, sigmoid_53
# x_se_52 => mean_13
# x_se_53 => convolution_132
# x_se_54 => mul_204, sigmoid_54
triton_poi_fused_convolution_mean_silu_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_95', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 132
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyrptw34e6d6cvsn76afxnxnk4ho6khj2zlgh3roszlve3wyb5w.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_341, x_342, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_55
# x_341 => mul_203, sigmoid_53
# x_342 => mul_205
# x_se_52 => mean_13
# x_se_53 => convolution_132
# x_se_54 => mul_204, sigmoid_54
# x_se_55 => convolution_133
triton_poi_fused_convolution_mean_mul_sigmoid_silu_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_96', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 620928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 1584
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehszw3hzcltas7j7drjlo6upxnxgdhhkb7rbc7uuzzpqco3gdgf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 => convolution_134
triton_poi_fused_convolution_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbywa7vkw5skcvk2sshs6zpgfi2274bxpl7o6k22mlnhh7oos7tw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 => convolution_135
triton_poi_fused_convolution_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxn6hbh3pnt5a6k5jpvrq2czbxyal6iu2xhw2q5jcfqe5owteipj.py
# Source Nodes: [cat_45, shortcut_17, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_45 => cat_36
# shortcut_17 => add_113
# x_345 => add_112, mul_207, mul_208, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_99', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 264
    x2 = xindex
    y1 = (yindex // 264)
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr0 + (y0 + (264*x2) + (12936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 132, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (6468*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 264, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-6468) + x2 + (49*y0) + (6468*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (264*x2) + (12936*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43nlv7wh36m6oaswczmnprdcjiq24hjlxi6e76suupigqo2f53w.py
# Source Nodes: [x_390, x_394, x_395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_390 => add_129, mul_236, mul_237, sub_57
# x_394 => relu_6
# x_395 => mean_16
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_100', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (192, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg10_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg11_1, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (40, ), (1, ))
    assert_size_stride(arg15_1, (40, ), (1, ))
    assert_size_stride(arg16_1, (120, ), (1, ))
    assert_size_stride(arg17_1, (120, ), (1, ))
    assert_size_stride(arg18_1, (120, ), (1, ))
    assert_size_stride(arg19_1, (120, ), (1, ))
    assert_size_stride(arg20_1, (40, ), (1, ))
    assert_size_stride(arg21_1, (40, ), (1, ))
    assert_size_stride(arg22_1, (240, ), (1, ))
    assert_size_stride(arg23_1, (240, ), (1, ))
    assert_size_stride(arg24_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg25_1, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg26_1, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg27_1, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg28_1, (240, ), (1, ))
    assert_size_stride(arg29_1, (240, ), (1, ))
    assert_size_stride(arg30_1, (56, ), (1, ))
    assert_size_stride(arg31_1, (56, ), (1, ))
    assert_size_stride(arg32_1, (336, ), (1, ))
    assert_size_stride(arg33_1, (336, ), (1, ))
    assert_size_stride(arg34_1, (336, ), (1, ))
    assert_size_stride(arg35_1, (336, ), (1, ))
    assert_size_stride(arg36_1, (56, ), (1, ))
    assert_size_stride(arg37_1, (56, ), (1, ))
    assert_size_stride(arg38_1, (336, ), (1, ))
    assert_size_stride(arg39_1, (336, ), (1, ))
    assert_size_stride(arg40_1, (336, ), (1, ))
    assert_size_stride(arg41_1, (336, ), (1, ))
    assert_size_stride(arg42_1, (56, ), (1, ))
    assert_size_stride(arg43_1, (56, ), (1, ))
    assert_size_stride(arg44_1, (336, ), (1, ))
    assert_size_stride(arg45_1, (336, ), (1, ))
    assert_size_stride(arg46_1, (336, ), (1, ))
    assert_size_stride(arg47_1, (336, ), (1, ))
    assert_size_stride(arg48_1, (56, ), (1, ))
    assert_size_stride(arg49_1, (56, ), (1, ))
    assert_size_stride(arg50_1, (336, ), (1, ))
    assert_size_stride(arg51_1, (336, ), (1, ))
    assert_size_stride(arg52_1, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg53_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg54_1, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg55_1, (336, ), (1, ))
    assert_size_stride(arg56_1, (336, ), (1, ))
    assert_size_stride(arg57_1, (104, ), (1, ))
    assert_size_stride(arg58_1, (104, ), (1, ))
    assert_size_stride(arg59_1, (624, ), (1, ))
    assert_size_stride(arg60_1, (624, ), (1, ))
    assert_size_stride(arg61_1, (624, ), (1, ))
    assert_size_stride(arg62_1, (624, ), (1, ))
    assert_size_stride(arg63_1, (104, ), (1, ))
    assert_size_stride(arg64_1, (104, ), (1, ))
    assert_size_stride(arg65_1, (624, ), (1, ))
    assert_size_stride(arg66_1, (624, ), (1, ))
    assert_size_stride(arg67_1, (624, ), (1, ))
    assert_size_stride(arg68_1, (624, ), (1, ))
    assert_size_stride(arg69_1, (104, ), (1, ))
    assert_size_stride(arg70_1, (104, ), (1, ))
    assert_size_stride(arg71_1, (624, ), (1, ))
    assert_size_stride(arg72_1, (624, ), (1, ))
    assert_size_stride(arg73_1, (624, ), (1, ))
    assert_size_stride(arg74_1, (624, ), (1, ))
    assert_size_stride(arg75_1, (104, ), (1, ))
    assert_size_stride(arg76_1, (104, ), (1, ))
    assert_size_stride(arg77_1, (624, ), (1, ))
    assert_size_stride(arg78_1, (624, ), (1, ))
    assert_size_stride(arg79_1, (624, ), (1, ))
    assert_size_stride(arg80_1, (624, ), (1, ))
    assert_size_stride(arg81_1, (160, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (480, ), (1, ))
    assert_size_stride(arg84_1, (480, ), (1, ))
    assert_size_stride(arg85_1, (480, ), (1, ))
    assert_size_stride(arg86_1, (480, ), (1, ))
    assert_size_stride(arg87_1, (160, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (480, ), (1, ))
    assert_size_stride(arg90_1, (480, ), (1, ))
    assert_size_stride(arg91_1, (480, ), (1, ))
    assert_size_stride(arg92_1, (480, ), (1, ))
    assert_size_stride(arg93_1, (160, ), (1, ))
    assert_size_stride(arg94_1, (160, ), (1, ))
    assert_size_stride(arg95_1, (480, ), (1, ))
    assert_size_stride(arg96_1, (480, ), (1, ))
    assert_size_stride(arg97_1, (480, ), (1, ))
    assert_size_stride(arg98_1, (480, ), (1, ))
    assert_size_stride(arg99_1, (160, ), (1, ))
    assert_size_stride(arg100_1, (160, ), (1, ))
    assert_size_stride(arg101_1, (960, ), (1, ))
    assert_size_stride(arg102_1, (960, ), (1, ))
    assert_size_stride(arg103_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg104_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg106_1, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg107_1, (960, ), (1, ))
    assert_size_stride(arg108_1, (960, ), (1, ))
    assert_size_stride(arg109_1, (264, ), (1, ))
    assert_size_stride(arg110_1, (264, ), (1, ))
    assert_size_stride(arg111_1, (1584, ), (1, ))
    assert_size_stride(arg112_1, (1584, ), (1, ))
    assert_size_stride(arg113_1, (1584, ), (1, ))
    assert_size_stride(arg114_1, (1584, ), (1, ))
    assert_size_stride(arg115_1, (264, ), (1, ))
    assert_size_stride(arg116_1, (264, ), (1, ))
    assert_size_stride(arg117_1, (1584, ), (1, ))
    assert_size_stride(arg118_1, (1584, ), (1, ))
    assert_size_stride(arg119_1, (1584, ), (1, ))
    assert_size_stride(arg120_1, (1584, ), (1, ))
    assert_size_stride(arg121_1, (264, ), (1, ))
    assert_size_stride(arg122_1, (264, ), (1, ))
    assert_size_stride(arg123_1, (1584, ), (1, ))
    assert_size_stride(arg124_1, (1584, ), (1, ))
    assert_size_stride(arg125_1, (1584, ), (1, ))
    assert_size_stride(arg126_1, (1584, ), (1, ))
    assert_size_stride(arg127_1, (264, ), (1, ))
    assert_size_stride(arg128_1, (264, ), (1, ))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (1536, ), (1, ))
    assert_size_stride(arg131_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg132_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg133_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg134_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg135_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg136_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg137_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg138_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg139_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg140_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg141_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg142_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg143_1, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg144_1, (20, ), (1, ))
    assert_size_stride(arg145_1, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg146_1, (240, ), (1, ))
    assert_size_stride(arg147_1, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg148_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg149_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg150_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg152_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg153_1, (28, ), (1, ))
    assert_size_stride(arg154_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg155_1, (336, ), (1, ))
    assert_size_stride(arg156_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg157_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg158_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg159_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg160_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg161_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg162_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg163_1, (28, ), (1, ))
    assert_size_stride(arg164_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg165_1, (336, ), (1, ))
    assert_size_stride(arg166_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg167_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg168_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg169_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg170_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg171_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg172_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg173_1, (28, ), (1, ))
    assert_size_stride(arg174_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg175_1, (336, ), (1, ))
    assert_size_stride(arg176_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg177_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg178_1, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg179_1, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg180_1, (14, ), (1, ))
    assert_size_stride(arg181_1, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg182_1, (336, ), (1, ))
    assert_size_stride(arg183_1, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg184_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg185_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg186_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg187_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg188_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg189_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg190_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg191_1, (26, ), (1, ))
    assert_size_stride(arg192_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg193_1, (624, ), (1, ))
    assert_size_stride(arg194_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg195_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg196_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg197_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg198_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg199_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg200_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg201_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg202_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg203_1, (26, ), (1, ))
    assert_size_stride(arg204_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg205_1, (624, ), (1, ))
    assert_size_stride(arg206_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg207_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg208_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg209_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg210_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg211_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg212_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg213_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg214_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg215_1, (26, ), (1, ))
    assert_size_stride(arg216_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg217_1, (624, ), (1, ))
    assert_size_stride(arg218_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg219_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg220_1, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg221_1, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg222_1, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg223_1, (52, ), (1, ))
    assert_size_stride(arg224_1, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg225_1, (624, ), (1, ))
    assert_size_stride(arg226_1, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg227_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg228_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg229_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg230_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg231_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg232_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg233_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg234_1, (80, ), (1, ))
    assert_size_stride(arg235_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg236_1, (480, ), (1, ))
    assert_size_stride(arg237_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg238_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg239_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg240_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg241_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg242_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg243_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg244_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg245_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg246_1, (80, ), (1, ))
    assert_size_stride(arg247_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg248_1, (480, ), (1, ))
    assert_size_stride(arg249_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg250_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg251_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg252_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg253_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg254_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg255_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg256_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg257_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg258_1, (80, ), (1, ))
    assert_size_stride(arg259_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg260_1, (480, ), (1, ))
    assert_size_stride(arg261_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg262_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg263_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg264_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg265_1, (80, ), (1, ))
    assert_size_stride(arg266_1, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg267_1, (960, ), (1, ))
    assert_size_stride(arg268_1, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg269_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg270_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg271_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg272_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg273_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg274_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg275_1, (132, ), (1, ))
    assert_size_stride(arg276_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg277_1, (1584, ), (1, ))
    assert_size_stride(arg278_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg279_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg280_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg281_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg282_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg283_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg284_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg285_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg286_1, (132, ), (1, ))
    assert_size_stride(arg287_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg288_1, (1584, ), (1, ))
    assert_size_stride(arg289_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg290_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg291_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg292_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg293_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg294_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg295_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg296_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg297_1, (132, ), (1, ))
    assert_size_stride(arg298_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg299_1, (1584, ), (1, ))
    assert_size_stride(arg300_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg301_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg302_1, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg303_1, (1000, 1536), (1536, 1))
    assert_size_stride(arg304_1, (1000, ), (1, ))
    assert_size_stride(arg305_1, (32, ), (1, ))
    assert_size_stride(arg306_1, (32, ), (1, ))
    assert_size_stride(arg307_1, (32, ), (1, ))
    assert_size_stride(arg308_1, (32, ), (1, ))
    assert_size_stride(arg309_1, (32, ), (1, ))
    assert_size_stride(arg310_1, (32, ), (1, ))
    assert_size_stride(arg311_1, (192, ), (1, ))
    assert_size_stride(arg312_1, (192, ), (1, ))
    assert_size_stride(arg313_1, (192, ), (1, ))
    assert_size_stride(arg314_1, (192, ), (1, ))
    assert_size_stride(arg315_1, (40, ), (1, ))
    assert_size_stride(arg316_1, (40, ), (1, ))
    assert_size_stride(arg317_1, (120, ), (1, ))
    assert_size_stride(arg318_1, (120, ), (1, ))
    assert_size_stride(arg319_1, (120, ), (1, ))
    assert_size_stride(arg320_1, (120, ), (1, ))
    assert_size_stride(arg321_1, (40, ), (1, ))
    assert_size_stride(arg322_1, (40, ), (1, ))
    assert_size_stride(arg323_1, (240, ), (1, ))
    assert_size_stride(arg324_1, (240, ), (1, ))
    assert_size_stride(arg325_1, (240, ), (1, ))
    assert_size_stride(arg326_1, (240, ), (1, ))
    assert_size_stride(arg327_1, (56, ), (1, ))
    assert_size_stride(arg328_1, (56, ), (1, ))
    assert_size_stride(arg329_1, (336, ), (1, ))
    assert_size_stride(arg330_1, (336, ), (1, ))
    assert_size_stride(arg331_1, (336, ), (1, ))
    assert_size_stride(arg332_1, (336, ), (1, ))
    assert_size_stride(arg333_1, (56, ), (1, ))
    assert_size_stride(arg334_1, (56, ), (1, ))
    assert_size_stride(arg335_1, (336, ), (1, ))
    assert_size_stride(arg336_1, (336, ), (1, ))
    assert_size_stride(arg337_1, (336, ), (1, ))
    assert_size_stride(arg338_1, (336, ), (1, ))
    assert_size_stride(arg339_1, (56, ), (1, ))
    assert_size_stride(arg340_1, (56, ), (1, ))
    assert_size_stride(arg341_1, (336, ), (1, ))
    assert_size_stride(arg342_1, (336, ), (1, ))
    assert_size_stride(arg343_1, (336, ), (1, ))
    assert_size_stride(arg344_1, (336, ), (1, ))
    assert_size_stride(arg345_1, (56, ), (1, ))
    assert_size_stride(arg346_1, (56, ), (1, ))
    assert_size_stride(arg347_1, (336, ), (1, ))
    assert_size_stride(arg348_1, (336, ), (1, ))
    assert_size_stride(arg349_1, (336, ), (1, ))
    assert_size_stride(arg350_1, (336, ), (1, ))
    assert_size_stride(arg351_1, (104, ), (1, ))
    assert_size_stride(arg352_1, (104, ), (1, ))
    assert_size_stride(arg353_1, (624, ), (1, ))
    assert_size_stride(arg354_1, (624, ), (1, ))
    assert_size_stride(arg355_1, (624, ), (1, ))
    assert_size_stride(arg356_1, (624, ), (1, ))
    assert_size_stride(arg357_1, (104, ), (1, ))
    assert_size_stride(arg358_1, (104, ), (1, ))
    assert_size_stride(arg359_1, (624, ), (1, ))
    assert_size_stride(arg360_1, (624, ), (1, ))
    assert_size_stride(arg361_1, (624, ), (1, ))
    assert_size_stride(arg362_1, (624, ), (1, ))
    assert_size_stride(arg363_1, (104, ), (1, ))
    assert_size_stride(arg364_1, (104, ), (1, ))
    assert_size_stride(arg365_1, (624, ), (1, ))
    assert_size_stride(arg366_1, (624, ), (1, ))
    assert_size_stride(arg367_1, (624, ), (1, ))
    assert_size_stride(arg368_1, (624, ), (1, ))
    assert_size_stride(arg369_1, (104, ), (1, ))
    assert_size_stride(arg370_1, (104, ), (1, ))
    assert_size_stride(arg371_1, (624, ), (1, ))
    assert_size_stride(arg372_1, (624, ), (1, ))
    assert_size_stride(arg373_1, (624, ), (1, ))
    assert_size_stride(arg374_1, (624, ), (1, ))
    assert_size_stride(arg375_1, (160, ), (1, ))
    assert_size_stride(arg376_1, (160, ), (1, ))
    assert_size_stride(arg377_1, (480, ), (1, ))
    assert_size_stride(arg378_1, (480, ), (1, ))
    assert_size_stride(arg379_1, (480, ), (1, ))
    assert_size_stride(arg380_1, (480, ), (1, ))
    assert_size_stride(arg381_1, (160, ), (1, ))
    assert_size_stride(arg382_1, (160, ), (1, ))
    assert_size_stride(arg383_1, (480, ), (1, ))
    assert_size_stride(arg384_1, (480, ), (1, ))
    assert_size_stride(arg385_1, (480, ), (1, ))
    assert_size_stride(arg386_1, (480, ), (1, ))
    assert_size_stride(arg387_1, (160, ), (1, ))
    assert_size_stride(arg388_1, (160, ), (1, ))
    assert_size_stride(arg389_1, (480, ), (1, ))
    assert_size_stride(arg390_1, (480, ), (1, ))
    assert_size_stride(arg391_1, (480, ), (1, ))
    assert_size_stride(arg392_1, (480, ), (1, ))
    assert_size_stride(arg393_1, (160, ), (1, ))
    assert_size_stride(arg394_1, (160, ), (1, ))
    assert_size_stride(arg395_1, (960, ), (1, ))
    assert_size_stride(arg396_1, (960, ), (1, ))
    assert_size_stride(arg397_1, (960, ), (1, ))
    assert_size_stride(arg398_1, (960, ), (1, ))
    assert_size_stride(arg399_1, (264, ), (1, ))
    assert_size_stride(arg400_1, (264, ), (1, ))
    assert_size_stride(arg401_1, (1584, ), (1, ))
    assert_size_stride(arg402_1, (1584, ), (1, ))
    assert_size_stride(arg403_1, (1584, ), (1, ))
    assert_size_stride(arg404_1, (1584, ), (1, ))
    assert_size_stride(arg405_1, (264, ), (1, ))
    assert_size_stride(arg406_1, (264, ), (1, ))
    assert_size_stride(arg407_1, (1584, ), (1, ))
    assert_size_stride(arg408_1, (1584, ), (1, ))
    assert_size_stride(arg409_1, (1584, ), (1, ))
    assert_size_stride(arg410_1, (1584, ), (1, ))
    assert_size_stride(arg411_1, (264, ), (1, ))
    assert_size_stride(arg412_1, (264, ), (1, ))
    assert_size_stride(arg413_1, (1584, ), (1, ))
    assert_size_stride(arg414_1, (1584, ), (1, ))
    assert_size_stride(arg415_1, (1584, ), (1, ))
    assert_size_stride(arg416_1, (1584, ), (1, ))
    assert_size_stride(arg417_1, (264, ), (1, ))
    assert_size_stride(arg418_1, (264, ), (1, ))
    assert_size_stride(arg419_1, (1536, ), (1, ))
    assert_size_stride(arg420_1, (1536, ), (1, ))
    assert_size_stride(arg421_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg421_1, buf0, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del arg421_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg305_1, arg306_1, arg1_1, arg2_1, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg1_1
        del arg2_1
        del arg305_1
        del arg306_1
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg131_1
        buf5 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, arg307_1, arg308_1, arg3_1, arg4_1, buf5, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg307_1
        del arg308_1
        del arg3_1
        del arg4_1
        del buf4
        # Source Nodes: [x_10, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg132_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [shortcut_1, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf7, arg309_1, arg310_1, arg5_1, arg6_1, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg309_1
        del arg310_1
        del arg5_1
        del arg6_1
        del buf3
        buf8 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf7, buf8, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg133_1
        buf10 = buf8; del buf8  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf7, buf10, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf7
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg134_1
        del buf10
        buf12 = empty((8, 192, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_81, x_20, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf9, buf11, arg311_1, arg312_1, arg7_1, arg8_1, buf12, 19267584, grid=grid(19267584), stream=stream0)
        del arg311_1
        del arg312_1
        del arg7_1
        del arg8_1
        del buf11
        del buf9
        buf13 = empty_strided((8, 64, 113, 113), (817216, 1, 7232, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_7.run(buf12, buf13, 512, 12769, grid=grid(512, 12769), stream=stream0)
        # Source Nodes: [conv2d_1, x_25], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg9_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf14, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg9_1
        del buf13
        buf15 = empty_strided((8, 64, 115, 115), (846400, 1, 7360, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_8.run(buf12, buf15, 512, 13225, grid=grid(512, 13225), stream=stream0)
        # Source Nodes: [conv2d_2, x_27], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg10_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf16, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg10_1
        del buf15
        buf17 = empty_strided((8, 64, 117, 117), (876096, 1, 7488, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_9.run(buf12, buf17, 512, 13689, grid=grid(512, 13689), stream=stream0)
        del buf12
        # Source Nodes: [conv2d_3, x_29], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf18, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg11_1
        del buf17
        buf19 = empty((8, 192, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_80, x_32, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10.run(buf14, buf16, buf18, arg313_1, arg314_1, arg12_1, arg13_1, buf19, 4816896, grid=grid(4816896), stream=stream0)
        del arg12_1
        del arg13_1
        del arg313_1
        del arg314_1
        del buf14
        del buf16
        del buf18
        buf20 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf19, buf20, 768, 3136, grid=grid(768, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 20, 56, 56), (62720, 3136, 56, 1))
        del arg135_1
        buf22 = buf20; del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf19, buf22, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del buf19
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 20, 56, 56), (62720, 3136, 56, 1))
        del arg136_1
        del buf22
        buf24 = empty((8, 40, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_79, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_13.run(buf21, buf23, arg315_1, arg316_1, arg14_1, arg15_1, buf24, 1003520, grid=grid(1003520), stream=stream0)
        del arg14_1
        del arg15_1
        del arg315_1
        del arg316_1
        del buf21
        buf25 = reinterpret_tensor(buf23, (8, 20, 56, 56), (62720, 1, 1120, 20), 0); del buf23  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf24, buf25, 160, 3136, grid=grid(160, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 60, 56, 56), (188160, 3136, 56, 1))
        del arg137_1
        buf27 = buf25; del buf25  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf24, buf27, 160, 3136, grid=grid(160, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 60, 56, 56), (188160, 3136, 56, 1))
        del arg138_1
        del buf27
        buf29 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_78, x_45, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf26, buf28, arg317_1, arg318_1, arg16_1, arg17_1, buf29, 960, 3136, grid=grid(960, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        del arg317_1
        del arg318_1
        # Source Nodes: [cat_78, x_45, x_48, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf30, (8, 120, 56, 56), (376320, 3136, 56, 1))
        del arg139_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf31, arg319_1, arg320_1, arg18_1, arg19_1, 3010560, grid=grid(3010560), stream=stream0)
        del arg18_1
        del arg19_1
        del arg319_1
        del arg320_1
        buf32 = reinterpret_tensor(buf28, (8, 60, 56, 56), (188160, 1, 3360, 60), 0); del buf28  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf31, buf32, 480, 3136, grid=grid(480, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 20, 56, 56), (62720, 3136, 56, 1))
        del arg140_1
        buf34 = buf32; del buf32  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf31, buf34, 480, 3136, grid=grid(480, 3136), stream=stream0)
        del buf31
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 20, 56, 56), (62720, 3136, 56, 1))
        del arg141_1
        buf36 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_77, shortcut_3, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_20.run(buf33, buf35, arg321_1, arg322_1, arg20_1, arg21_1, buf24, buf36, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del arg20_1
        del arg21_1
        del arg321_1
        del arg322_1
        del buf24
        del buf33
        del buf35
        # Source Nodes: [cat_77, shortcut_3, x_57, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 240, 56, 56), (752640, 3136, 56, 1))
        del arg142_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf38, arg323_1, arg324_1, arg22_1, arg23_1, 6021120, grid=grid(6021120), stream=stream0)
        del arg22_1
        del arg23_1
        del arg323_1
        del arg324_1
        buf39 = empty_strided((8, 60, 57, 57), (194940, 1, 3420, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_22.run(buf38, buf39, 480, 3249, grid=grid(480, 3249), stream=stream0)
        # Source Nodes: [conv2d_4, x_68], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg24_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf40, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg24_1
        del buf39
        buf41 = empty_strided((8, 60, 59, 59), (208860, 1, 3540, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_23.run(buf38, buf41, 480, 3481, grid=grid(480, 3481), stream=stream0)
        # Source Nodes: [conv2d_5, x_70], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg25_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf42, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg25_1
        del buf41
        buf43 = empty_strided((8, 60, 61, 61), (223260, 1, 3660, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_24.run(buf38, buf43, 480, 3721, grid=grid(480, 3721), stream=stream0)
        # Source Nodes: [conv2d_6, x_72], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg26_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf44, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg26_1
        del buf43
        buf45 = empty_strided((8, 60, 63, 63), (238140, 1, 3780, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_25.run(buf38, buf45, 480, 3969, grid=grid(480, 3969), stream=stream0)
        del buf38
        # Source Nodes: [conv2d_7, x_74], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg27_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf46, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg27_1
        del buf45
        buf47 = reinterpret_tensor(buf34, (8, 240, 28, 28), (188160, 784, 28, 1), 0); del buf34  # reuse
        buf48 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf49 = reinterpret_tensor(buf48, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf48  # reuse
        # Source Nodes: [cat_76, x_77, x_80, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_26.run(buf49, buf40, buf42, buf44, buf46, arg325_1, arg326_1, arg28_1, arg29_1, buf47, 1920, 784, grid=grid(1920), stream=stream0)
        del arg28_1
        del arg29_1
        del arg325_1
        del arg326_1
        del buf40
        del buf42
        del buf44
        del buf46
        # Source Nodes: [x_80, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf50 = extern_kernels.convolution(buf49, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg143_1
        del buf49
        buf51 = reinterpret_tensor(buf50, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf50  # reuse
        # Source Nodes: [x_80, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_27.run(buf51, arg144_1, 160, grid=grid(160), stream=stream0)
        del arg144_1
        # Source Nodes: [x_80, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf52 = extern_kernels.convolution(buf51, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg145_1
        del buf51
        buf53 = reinterpret_tensor(buf26, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf26  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_28.run(buf47, buf52, arg146_1, buf53, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg146_1
        del buf47
        del buf52
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81, x_82, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf54 = extern_kernels.convolution(buf53, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 56, 28, 28), (43904, 784, 28, 1))
        del arg147_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf55, arg327_1, arg328_1, arg30_1, arg31_1, 351232, grid=grid(351232), stream=stream0)
        del arg30_1
        del arg31_1
        del arg327_1
        del arg328_1
        buf56 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf55, buf56, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg148_1
        buf58 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf55, buf58, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg149_1
        del buf58
        buf60 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [cat_75, x_89, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_32.run(buf61, buf57, buf59, arg329_1, arg330_1, arg32_1, arg33_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg329_1
        del arg32_1
        del arg330_1
        del arg33_1
        del buf57
        buf62 = reinterpret_tensor(buf59, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf59  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf61, buf62, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf63, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg150_1
        buf64 = buf62; del buf62  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf61, buf64, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg151_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf65, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg151_1
        del buf64
        buf66 = buf61; del buf61  # reuse
        buf67 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf68 = reinterpret_tensor(buf67, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf67  # reuse
        # Source Nodes: [cat_74, x_95, x_98, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_35.run(buf68, buf63, buf65, arg331_1, arg332_1, arg34_1, arg35_1, buf66, 2688, 784, grid=grid(2688), stream=stream0)
        del arg331_1
        del arg332_1
        del arg34_1
        del arg35_1
        del buf63
        # Source Nodes: [x_98, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf69 = extern_kernels.convolution(buf68, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg152_1
        del buf68
        buf70 = reinterpret_tensor(buf69, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf69  # reuse
        # Source Nodes: [x_98, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf70, arg153_1, 224, grid=grid(224), stream=stream0)
        del arg153_1
        # Source Nodes: [x_98, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf71 = extern_kernels.convolution(buf70, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg154_1
        del buf70
        buf72 = buf66; del buf66  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_98, x_99, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf72, buf71, arg155_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg155_1
        buf73 = reinterpret_tensor(buf65, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf65  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf72, buf73, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg156_1
        buf75 = buf73; del buf73  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf72, buf75, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg157_1
        del buf75
        buf77 = buf55; del buf55  # reuse
        # Source Nodes: [cat_73, shortcut_5, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_38.run(buf77, buf74, buf76, arg333_1, arg334_1, arg36_1, arg37_1, 351232, grid=grid(351232), stream=stream0)
        del arg333_1
        del arg334_1
        del arg36_1
        del arg37_1
        del buf74
        buf78 = reinterpret_tensor(buf76, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf76  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf77, buf78, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg158_1
        buf80 = buf78; del buf78  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf77, buf80, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg159_1
        del buf80
        buf82 = buf72; del buf72  # reuse
        buf83 = buf82; del buf82  # reuse
        # Source Nodes: [cat_72, x_109, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_32.run(buf83, buf79, buf81, arg335_1, arg336_1, arg38_1, arg39_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg335_1
        del arg336_1
        del arg38_1
        del arg39_1
        del buf79
        buf84 = reinterpret_tensor(buf81, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf81  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf83, buf84, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg160_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf85, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg160_1
        buf86 = buf84; del buf84  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf83, buf86, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg161_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf87, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg161_1
        del buf86
        buf88 = buf83; del buf83  # reuse
        buf89 = reinterpret_tensor(buf71, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf71  # reuse
        buf90 = reinterpret_tensor(buf89, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf89  # reuse
        # Source Nodes: [cat_71, x_115, x_118, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_35.run(buf90, buf85, buf87, arg337_1, arg338_1, arg40_1, arg41_1, buf88, 2688, 784, grid=grid(2688), stream=stream0)
        del arg337_1
        del arg338_1
        del arg40_1
        del arg41_1
        del buf85
        # Source Nodes: [x_118, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf91 = extern_kernels.convolution(buf90, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg162_1
        del buf90
        buf92 = reinterpret_tensor(buf91, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf91  # reuse
        # Source Nodes: [x_118, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf92, arg163_1, 224, grid=grid(224), stream=stream0)
        del arg163_1
        # Source Nodes: [x_118, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf93 = extern_kernels.convolution(buf92, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg164_1
        del buf92
        buf94 = buf88; del buf88  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_118, x_119, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf94, buf93, arg165_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg165_1
        buf95 = reinterpret_tensor(buf87, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf87  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf94, buf95, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg166_1
        buf97 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf94, buf97, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg167_1
        del buf97
        buf99 = buf77; del buf77  # reuse
        # Source Nodes: [cat_70, shortcut_6, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_38.run(buf99, buf96, buf98, arg339_1, arg340_1, arg42_1, arg43_1, 351232, grid=grid(351232), stream=stream0)
        del arg339_1
        del arg340_1
        del arg42_1
        del arg43_1
        del buf96
        buf100 = reinterpret_tensor(buf98, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf98  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf99, buf100, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg168_1
        buf102 = buf100; del buf100  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf99, buf102, 224, 784, grid=grid(224, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg169_1
        del buf102
        buf104 = buf94; del buf94  # reuse
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [cat_69, x_129, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_32.run(buf105, buf101, buf103, arg341_1, arg342_1, arg44_1, arg45_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg341_1
        del arg342_1
        del arg44_1
        del arg45_1
        del buf101
        buf106 = reinterpret_tensor(buf103, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf103  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf105, buf106, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf107, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg170_1
        buf108 = buf106; del buf106  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf105, buf108, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg171_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf109, (8, 168, 28, 28), (131712, 784, 28, 1))
        del arg171_1
        del buf108
        buf110 = buf105; del buf105  # reuse
        buf111 = reinterpret_tensor(buf93, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf93  # reuse
        buf112 = reinterpret_tensor(buf111, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf111  # reuse
        # Source Nodes: [cat_68, x_135, x_138, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_35.run(buf112, buf107, buf109, arg343_1, arg344_1, arg46_1, arg47_1, buf110, 2688, 784, grid=grid(2688), stream=stream0)
        del arg343_1
        del arg344_1
        del arg46_1
        del arg47_1
        del buf107
        # Source Nodes: [x_138, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf113 = extern_kernels.convolution(buf112, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg172_1
        del buf112
        buf114 = reinterpret_tensor(buf113, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf113  # reuse
        # Source Nodes: [x_138, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf114, arg173_1, 224, grid=grid(224), stream=stream0)
        del arg173_1
        # Source Nodes: [x_138, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf115 = extern_kernels.convolution(buf114, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg174_1
        del buf114
        buf116 = buf110; del buf110  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_138, x_139, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf116, buf115, arg175_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg175_1
        buf117 = reinterpret_tensor(buf109, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf109  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf116, buf117, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg176_1
        buf119 = buf117; del buf117  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf116, buf119, 1344, 784, grid=grid(1344, 784), stream=stream0)
        del buf116
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 28, 28, 28), (21952, 784, 28, 1))
        del arg177_1
        del buf119
        buf121 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_67, shortcut_7, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_39.run(buf118, buf120, arg345_1, arg346_1, arg48_1, arg49_1, buf99, buf121, 448, 784, grid=grid(448, 784), stream=stream0)
        del arg345_1
        del arg346_1
        del arg48_1
        del arg49_1
        del buf118
        del buf120
        del buf99
        # Source Nodes: [cat_67, shortcut_7, x_142, x_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 336, 28, 28), (263424, 784, 28, 1))
        del arg178_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_40.run(buf123, arg347_1, arg348_1, arg50_1, arg51_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg347_1
        del arg348_1
        del arg50_1
        del arg51_1
        buf124 = empty_strided((8, 112, 29, 29), (94192, 1, 3248, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_41.run(buf123, buf124, 896, 841, grid=grid(896, 841), stream=stream0)
        # Source Nodes: [conv2d_8, x_153], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg52_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf125, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg52_1
        del buf124
        buf126 = empty_strided((8, 112, 31, 31), (107632, 1, 3472, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_42.run(buf123, buf126, 896, 961, grid=grid(896, 961), stream=stream0)
        # Source Nodes: [conv2d_9, x_155], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg53_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf127, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg53_1
        del buf126
        buf128 = empty_strided((8, 112, 33, 33), (121968, 1, 3696, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_43.run(buf123, buf128, 896, 1089, grid=grid(896, 1089), stream=stream0)
        del buf123
        # Source Nodes: [conv2d_10, x_157], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg54_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf129, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg54_1
        del buf128
        buf130 = empty((8, 336, 14, 14), device='cuda', dtype=torch.float32)
        buf131 = reinterpret_tensor(buf115, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf115  # reuse
        buf132 = reinterpret_tensor(buf131, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf131  # reuse
        # Source Nodes: [cat_66, x_160, x_163, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_44.run(buf132, buf125, buf127, buf129, arg349_1, arg350_1, arg55_1, arg56_1, buf130, 2688, 196, grid=grid(2688), stream=stream0)
        del arg349_1
        del arg350_1
        del arg55_1
        del arg56_1
        del buf125
        del buf127
        del buf129
        # Source Nodes: [x_163, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf133 = extern_kernels.convolution(buf132, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 14, 1, 1), (14, 1, 1, 1))
        del arg179_1
        del buf132
        buf134 = reinterpret_tensor(buf133, (8, 14, 1, 1), (14, 1, 14, 14), 0); del buf133  # reuse
        # Source Nodes: [x_163, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_45.run(buf134, arg180_1, 112, grid=grid(112), stream=stream0)
        del arg180_1
        # Source Nodes: [x_163, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf135 = extern_kernels.convolution(buf134, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg181_1
        del buf134
        buf136 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_46.run(buf130, buf135, arg182_1, buf136, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg182_1
        del buf130
        del buf135
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164, x_165, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf137 = extern_kernels.convolution(buf136, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 104, 14, 14), (20384, 196, 14, 1))
        del arg183_1
        del buf136
        buf138 = buf137; del buf137  # reuse
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf138, arg351_1, arg352_1, arg57_1, arg58_1, 163072, grid=grid(163072), stream=stream0)
        del arg351_1
        del arg352_1
        del arg57_1
        del arg58_1
        buf139 = empty_strided((8, 52, 14, 14), (10192, 1, 728, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf138, buf139, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg184_1
        buf141 = buf139; del buf139  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf138, buf141, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg185_1
        del buf141
        buf143 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [cat_65, x_172, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_50.run(buf144, buf140, buf142, arg353_1, arg354_1, arg59_1, arg60_1, 978432, grid=grid(978432), stream=stream0)
        del arg353_1
        del arg354_1
        del arg59_1
        del arg60_1
        del buf140
        buf145 = empty_strided((8, 156, 14, 14), (30576, 1, 2184, 156), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf144, buf145, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf146, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg186_1
        buf147 = buf145; del buf145  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf144, buf147, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg187_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf148, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg187_1
        buf149 = buf147; del buf147  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf144, buf149, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg188_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf150, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg188_1
        buf151 = buf149; del buf149  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf144, buf151, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg189_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf152, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg189_1
        del buf151
        buf153 = buf144; del buf144  # reuse
        buf154 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf155 = reinterpret_tensor(buf154, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf154  # reuse
        # Source Nodes: [cat_64, x_178, x_181, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55.run(buf155, buf146, buf148, buf150, buf152, arg355_1, arg356_1, arg61_1, arg62_1, buf153, 4992, 196, grid=grid(4992), stream=stream0)
        del arg355_1
        del arg356_1
        del arg61_1
        del arg62_1
        del buf146
        del buf148
        del buf150
        # Source Nodes: [x_181, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf156 = extern_kernels.convolution(buf155, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg190_1
        del buf155
        buf157 = reinterpret_tensor(buf156, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf156  # reuse
        # Source Nodes: [x_181, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_56.run(buf157, arg191_1, 208, grid=grid(208), stream=stream0)
        del arg191_1
        # Source Nodes: [x_181, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf158 = extern_kernels.convolution(buf157, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg192_1
        del buf157
        buf159 = buf153; del buf153  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_181, x_182, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_57.run(buf159, buf158, arg193_1, 978432, grid=grid(978432), stream=stream0)
        del arg193_1
        buf160 = reinterpret_tensor(buf142, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf142  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf159, buf160, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg194_1
        buf162 = buf160; del buf160  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf159, buf162, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg195_1
        del buf162
        buf164 = buf138; del buf138  # reuse
        # Source Nodes: [cat_63, shortcut_9, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_60.run(buf164, buf161, buf163, arg357_1, arg358_1, arg63_1, arg64_1, 163072, grid=grid(163072), stream=stream0)
        del arg357_1
        del arg358_1
        del arg63_1
        del arg64_1
        del buf161
        buf165 = reinterpret_tensor(buf163, (8, 52, 14, 14), (10192, 1, 728, 52), 0); del buf163  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf164, buf165, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg196_1
        buf167 = buf165; del buf165  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf164, buf167, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg197_1
        del buf167
        buf169 = buf159; del buf159  # reuse
        buf170 = buf169; del buf169  # reuse
        # Source Nodes: [cat_62, x_192, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_50.run(buf170, buf166, buf168, arg359_1, arg360_1, arg65_1, arg66_1, 978432, grid=grid(978432), stream=stream0)
        del arg359_1
        del arg360_1
        del arg65_1
        del arg66_1
        del buf166
        buf171 = reinterpret_tensor(buf152, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf152  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf170, buf171, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg198_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf172, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg198_1
        buf173 = buf171; del buf171  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf170, buf173, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg199_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf174, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg199_1
        buf175 = buf173; del buf173  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf170, buf175, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg200_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf176, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg200_1
        buf177 = buf175; del buf175  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf170, buf177, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg201_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf178, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg201_1
        del buf177
        buf179 = buf170; del buf170  # reuse
        buf180 = reinterpret_tensor(buf158, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf158  # reuse
        buf181 = reinterpret_tensor(buf180, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf180  # reuse
        # Source Nodes: [cat_61, x_198, x_201, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55.run(buf181, buf172, buf174, buf176, buf178, arg361_1, arg362_1, arg67_1, arg68_1, buf179, 4992, 196, grid=grid(4992), stream=stream0)
        del arg361_1
        del arg362_1
        del arg67_1
        del arg68_1
        del buf172
        del buf174
        del buf176
        # Source Nodes: [x_201, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf182 = extern_kernels.convolution(buf181, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg202_1
        del buf181
        buf183 = reinterpret_tensor(buf182, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf182  # reuse
        # Source Nodes: [x_201, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_56.run(buf183, arg203_1, 208, grid=grid(208), stream=stream0)
        del arg203_1
        # Source Nodes: [x_201, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf184 = extern_kernels.convolution(buf183, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg204_1
        del buf183
        buf185 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_201, x_202, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_57.run(buf185, buf184, arg205_1, 978432, grid=grid(978432), stream=stream0)
        del arg205_1
        buf186 = reinterpret_tensor(buf168, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf168  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf185, buf186, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg206_1
        buf188 = buf186; del buf186  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf185, buf188, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg207_1
        del buf188
        buf190 = buf164; del buf164  # reuse
        # Source Nodes: [cat_60, shortcut_10, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_60.run(buf190, buf187, buf189, arg363_1, arg364_1, arg69_1, arg70_1, 163072, grid=grid(163072), stream=stream0)
        del arg363_1
        del arg364_1
        del arg69_1
        del arg70_1
        del buf187
        buf191 = reinterpret_tensor(buf189, (8, 52, 14, 14), (10192, 1, 728, 52), 0); del buf189  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf190, buf191, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg208_1
        buf193 = buf191; del buf191  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf190, buf193, 416, 196, grid=grid(416, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 312, 14, 14), (61152, 196, 14, 1))
        del arg209_1
        del buf193
        buf195 = buf185; del buf185  # reuse
        buf196 = buf195; del buf195  # reuse
        # Source Nodes: [cat_59, x_212, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_50.run(buf196, buf192, buf194, arg365_1, arg366_1, arg71_1, arg72_1, 978432, grid=grid(978432), stream=stream0)
        del arg365_1
        del arg366_1
        del arg71_1
        del arg72_1
        del buf192
        buf197 = reinterpret_tensor(buf178, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf178  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf196, buf197, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg210_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf198, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg210_1
        buf199 = buf197; del buf197  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf196, buf199, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg211_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf200, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg211_1
        buf201 = buf199; del buf199  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf196, buf201, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg212_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf202, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg212_1
        buf203 = buf201; del buf201  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf196, buf203, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg213_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf204, (8, 156, 14, 14), (30576, 196, 14, 1))
        del arg213_1
        del buf203
        buf205 = buf196; del buf196  # reuse
        buf206 = reinterpret_tensor(buf184, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf184  # reuse
        buf207 = reinterpret_tensor(buf206, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf206  # reuse
        # Source Nodes: [cat_58, x_218, x_221, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55.run(buf207, buf198, buf200, buf202, buf204, arg367_1, arg368_1, arg73_1, arg74_1, buf205, 4992, 196, grid=grid(4992), stream=stream0)
        del arg367_1
        del arg368_1
        del arg73_1
        del arg74_1
        del buf198
        del buf200
        del buf202
        del buf204
        # Source Nodes: [x_221, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf208 = extern_kernels.convolution(buf207, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg214_1
        del buf207
        buf209 = reinterpret_tensor(buf208, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf208  # reuse
        # Source Nodes: [x_221, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_56.run(buf209, arg215_1, 208, grid=grid(208), stream=stream0)
        del arg215_1
        # Source Nodes: [x_221, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf210 = extern_kernels.convolution(buf209, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg216_1
        del buf209
        buf211 = buf205; del buf205  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_221, x_222, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_57.run(buf211, buf210, arg217_1, 978432, grid=grid(978432), stream=stream0)
        del arg217_1
        buf212 = reinterpret_tensor(buf194, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf194  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf211, buf212, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg218_1
        buf214 = buf212; del buf212  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf211, buf214, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 52, 14, 14), (10192, 196, 14, 1))
        del arg219_1
        del buf214
        buf216 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_57, shortcut_11, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_61.run(buf213, buf215, arg369_1, arg370_1, arg75_1, arg76_1, buf190, buf216, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg369_1
        del arg370_1
        del arg75_1
        del arg76_1
        del buf190
        del buf213
        del buf215
        # Source Nodes: [cat_57, shortcut_11, x_225, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 624, 14, 14), (122304, 196, 14, 1))
        del arg220_1
        del buf216
        buf218 = buf217; del buf217  # reuse
        buf219 = reinterpret_tensor(buf211, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf211  # reuse
        # Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_62.run(buf218, arg371_1, arg372_1, arg77_1, arg78_1, buf219, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del arg371_1
        del arg372_1
        del arg77_1
        del arg78_1
        del buf218
        # Source Nodes: [x_234, x_235], Original ATen: [aten.convolution, aten.silu]
        buf220 = extern_kernels.convolution(buf219, arg221_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
        assert_size_stride(buf220, (8, 624, 14, 14), (122304, 196, 14, 1))
        del arg221_1
        buf221 = buf220; del buf220  # reuse
        buf222 = reinterpret_tensor(buf210, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf210  # reuse
        buf223 = reinterpret_tensor(buf222, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf222  # reuse
        # Source Nodes: [x_236, x_239, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_63.run(buf221, buf223, arg373_1, arg374_1, arg79_1, arg80_1, 4992, 196, grid=grid(4992), stream=stream0)
        del arg373_1
        del arg374_1
        del arg79_1
        del arg80_1
        # Source Nodes: [x_239, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf224 = extern_kernels.convolution(buf223, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 52, 1, 1), (52, 1, 1, 1))
        del arg222_1
        del buf223
        buf225 = reinterpret_tensor(buf224, (8, 52, 1, 1), (52, 1, 52, 52), 0); del buf224  # reuse
        # Source Nodes: [x_239, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_64.run(buf225, arg223_1, 416, grid=grid(416), stream=stream0)
        del arg223_1
        # Source Nodes: [x_239, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf226 = extern_kernels.convolution(buf225, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg224_1
        del buf225
        buf227 = buf219; del buf219  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_65.run(buf221, buf226, arg225_1, buf227, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del arg225_1
        del buf221
        del buf226
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240, x_241, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf228 = extern_kernels.convolution(buf227, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 160, 14, 14), (31360, 196, 14, 1))
        del arg226_1
        del buf227
        buf229 = buf228; del buf228  # reuse
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_66.run(buf229, arg375_1, arg376_1, arg81_1, arg82_1, 250880, grid=grid(250880), stream=stream0)
        del arg375_1
        del arg376_1
        del arg81_1
        del arg82_1
        buf230 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf229, buf230, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg227_1
        buf232 = buf230; del buf230  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_68.run(buf229, buf232, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg228_1
        del buf232
        buf234 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        buf235 = buf234; del buf234  # reuse
        # Source Nodes: [cat_56, x_248, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_69.run(buf235, buf231, buf233, arg377_1, arg378_1, arg83_1, arg84_1, 752640, grid=grid(752640), stream=stream0)
        del arg377_1
        del arg378_1
        del arg83_1
        del arg84_1
        del buf231
        buf236 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf235, buf236, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg229_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf237, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg229_1
        buf238 = buf236; del buf236  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_71.run(buf235, buf238, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg230_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf239, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg230_1
        buf240 = buf238; del buf238  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf235, buf240, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg231_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf241, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg231_1
        buf242 = buf240; del buf240  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf235, buf242, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg232_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf243, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg232_1
        del buf242
        buf244 = buf235; del buf235  # reuse
        buf245 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf246 = reinterpret_tensor(buf245, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf245  # reuse
        # Source Nodes: [cat_55, x_254, x_257, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_74.run(buf246, buf237, buf239, buf241, buf243, arg379_1, arg380_1, arg85_1, arg86_1, buf244, 3840, 196, grid=grid(3840), stream=stream0)
        del arg379_1
        del arg380_1
        del arg85_1
        del arg86_1
        del buf237
        del buf239
        del buf241
        # Source Nodes: [x_257, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf247 = extern_kernels.convolution(buf246, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg233_1
        del buf246
        buf248 = reinterpret_tensor(buf247, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf247  # reuse
        # Source Nodes: [x_257, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_75.run(buf248, arg234_1, 640, grid=grid(640), stream=stream0)
        del arg234_1
        # Source Nodes: [x_257, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf249 = extern_kernels.convolution(buf248, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg235_1
        del buf248
        buf250 = buf244; del buf244  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_257, x_258, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_76.run(buf250, buf249, arg236_1, 752640, grid=grid(752640), stream=stream0)
        del arg236_1
        buf251 = reinterpret_tensor(buf233, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf233  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf250, buf251, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg237_1
        buf253 = buf251; del buf251  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_78.run(buf250, buf253, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg238_1
        del buf253
        buf255 = buf229; del buf229  # reuse
        # Source Nodes: [cat_54, shortcut_13, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_79.run(buf255, buf252, buf254, arg381_1, arg382_1, arg87_1, arg88_1, 250880, grid=grid(250880), stream=stream0)
        del arg381_1
        del arg382_1
        del arg87_1
        del arg88_1
        del buf252
        buf256 = reinterpret_tensor(buf254, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf254  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf255, buf256, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg239_1
        buf258 = buf256; del buf256  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_68.run(buf255, buf258, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg240_1
        del buf258
        buf260 = buf250; del buf250  # reuse
        buf261 = buf260; del buf260  # reuse
        # Source Nodes: [cat_53, x_268, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_69.run(buf261, buf257, buf259, arg383_1, arg384_1, arg89_1, arg90_1, 752640, grid=grid(752640), stream=stream0)
        del arg383_1
        del arg384_1
        del arg89_1
        del arg90_1
        del buf257
        buf262 = reinterpret_tensor(buf243, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf243  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf261, buf262, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf263, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg241_1
        buf264 = buf262; del buf262  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_71.run(buf261, buf264, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg242_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf265, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg242_1
        buf266 = buf264; del buf264  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf261, buf266, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, arg243_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf267, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg243_1
        buf268 = buf266; del buf266  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf261, buf268, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, arg244_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf269, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg244_1
        del buf268
        buf270 = buf261; del buf261  # reuse
        buf271 = reinterpret_tensor(buf249, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf249  # reuse
        buf272 = reinterpret_tensor(buf271, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf271  # reuse
        # Source Nodes: [cat_52, x_274, x_277, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_74.run(buf272, buf263, buf265, buf267, buf269, arg385_1, arg386_1, arg91_1, arg92_1, buf270, 3840, 196, grid=grid(3840), stream=stream0)
        del arg385_1
        del arg386_1
        del arg91_1
        del arg92_1
        del buf263
        del buf265
        del buf267
        # Source Nodes: [x_277, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf273 = extern_kernels.convolution(buf272, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg245_1
        del buf272
        buf274 = reinterpret_tensor(buf273, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf273  # reuse
        # Source Nodes: [x_277, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_75.run(buf274, arg246_1, 640, grid=grid(640), stream=stream0)
        del arg246_1
        # Source Nodes: [x_277, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf275 = extern_kernels.convolution(buf274, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg247_1
        del buf274
        buf276 = buf270; del buf270  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_277, x_278, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_76.run(buf276, buf275, arg248_1, 752640, grid=grid(752640), stream=stream0)
        del arg248_1
        buf277 = reinterpret_tensor(buf259, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf259  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf276, buf277, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg249_1
        buf279 = buf277; del buf277  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_78.run(buf276, buf279, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg250_1
        del buf279
        buf281 = buf255; del buf255  # reuse
        # Source Nodes: [cat_51, shortcut_14, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_79.run(buf281, buf278, buf280, arg387_1, arg388_1, arg93_1, arg94_1, 250880, grid=grid(250880), stream=stream0)
        del arg387_1
        del arg388_1
        del arg93_1
        del arg94_1
        del buf278
        buf282 = reinterpret_tensor(buf280, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf280  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf281, buf282, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg251_1
        buf284 = buf282; del buf282  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_68.run(buf281, buf284, 640, 196, grid=grid(640, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg252_1
        del buf284
        buf286 = buf276; del buf276  # reuse
        buf287 = buf286; del buf286  # reuse
        # Source Nodes: [cat_50, x_288, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_69.run(buf287, buf283, buf285, arg389_1, arg390_1, arg95_1, arg96_1, 752640, grid=grid(752640), stream=stream0)
        del arg389_1
        del arg390_1
        del arg95_1
        del arg96_1
        buf288 = reinterpret_tensor(buf269, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf269  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf287, buf288, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, arg253_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf289, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg253_1
        buf290 = buf288; del buf288  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_71.run(buf287, buf290, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg254_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf291, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg254_1
        buf292 = buf290; del buf290  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf287, buf292, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, arg255_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf293, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg255_1
        buf294 = buf292; del buf292  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf287, buf294, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, arg256_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf295, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg256_1
        del buf294
        buf296 = buf287; del buf287  # reuse
        buf297 = reinterpret_tensor(buf275, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf275  # reuse
        buf298 = reinterpret_tensor(buf297, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf297  # reuse
        # Source Nodes: [cat_49, x_294, x_297, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_74.run(buf298, buf289, buf291, buf293, buf295, arg391_1, arg392_1, arg97_1, arg98_1, buf296, 3840, 196, grid=grid(3840), stream=stream0)
        del arg391_1
        del arg392_1
        del arg97_1
        del arg98_1
        del buf289
        del buf291
        del buf293
        del buf295
        # Source Nodes: [x_297, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf299 = extern_kernels.convolution(buf298, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg257_1
        del buf298
        buf300 = reinterpret_tensor(buf299, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf299  # reuse
        # Source Nodes: [x_297, x_se_44, x_se_45, x_se_46], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_75.run(buf300, arg258_1, 640, grid=grid(640), stream=stream0)
        del arg258_1
        # Source Nodes: [x_297, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf301 = extern_kernels.convolution(buf300, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg259_1
        del buf300
        buf302 = buf296; del buf296  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_297, x_298, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_76.run(buf302, buf301, arg260_1, 752640, grid=grid(752640), stream=stream0)
        del arg260_1
        del buf301
        buf303 = reinterpret_tensor(buf285, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf285  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf302, buf303, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg261_1
        buf305 = buf303; del buf303  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_78.run(buf302, buf305, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del buf302
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg262_1
        buf307 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_48, shortcut_15, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_80.run(buf304, buf306, arg393_1, arg394_1, arg99_1, arg100_1, buf281, buf307, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del arg100_1
        del arg393_1
        del arg394_1
        del arg99_1
        del buf281
        del buf304
        del buf306
        # Source Nodes: [cat_48, shortcut_15, x_301, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
        buf308 = extern_kernels.convolution(buf307, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 960, 14, 14), (188160, 196, 14, 1))
        del arg263_1
        del buf307
        buf309 = buf308; del buf308  # reuse
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_81.run(buf309, arg395_1, arg396_1, arg101_1, arg102_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg101_1
        del arg102_1
        del arg395_1
        del arg396_1
        buf310 = empty_strided((8, 240, 15, 15), (54000, 1, 3600, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_82.run(buf309, buf310, 1920, 225, grid=grid(1920, 225), stream=stream0)
        # Source Nodes: [conv2d_11, x_312], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf311 = extern_kernels.convolution(buf310, arg103_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf311, (8, 240, 7, 7), (11760, 49, 7, 1))
        del arg103_1
        del buf310
        buf312 = empty_strided((8, 240, 17, 17), (69360, 1, 4080, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_83.run(buf309, buf312, 1920, 289, grid=grid(1920, 289), stream=stream0)
        # Source Nodes: [conv2d_12, x_314], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf313 = extern_kernels.convolution(buf312, arg104_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf313, (8, 240, 7, 7), (11760, 49, 7, 1))
        del arg104_1
        del buf312
        buf314 = empty_strided((8, 240, 19, 19), (86640, 1, 4560, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_84.run(buf309, buf314, 1920, 361, grid=grid(1920, 361), stream=stream0)
        # Source Nodes: [conv2d_13, x_316], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg105_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf315, (8, 240, 7, 7), (11760, 49, 7, 1))
        del arg105_1
        del buf314
        buf316 = empty_strided((8, 240, 21, 21), (105840, 1, 5040, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_85.run(buf309, buf316, 1920, 441, grid=grid(1920, 441), stream=stream0)
        del buf309
        # Source Nodes: [conv2d_14, x_318], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf317 = extern_kernels.convolution(buf316, arg106_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf317, (8, 240, 7, 7), (11760, 49, 7, 1))
        del arg106_1
        del buf316
        buf318 = reinterpret_tensor(buf305, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf305  # reuse
        buf319 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cuda', dtype=torch.float32)
        buf320 = reinterpret_tensor(buf319, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf319  # reuse
        # Source Nodes: [cat_47, x_321, x_324, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_86.run(buf320, buf311, buf313, buf315, buf317, arg397_1, arg398_1, arg107_1, arg108_1, buf318, 7680, 49, grid=grid(7680), stream=stream0)
        del arg107_1
        del arg108_1
        del arg397_1
        del arg398_1
        del buf311
        del buf313
        del buf315
        del buf317
        # Source Nodes: [x_324, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf321 = extern_kernels.convolution(buf320, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg264_1
        del buf320
        buf322 = reinterpret_tensor(buf321, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf321  # reuse
        # Source Nodes: [x_324, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_75.run(buf322, arg265_1, 640, grid=grid(640), stream=stream0)
        del arg265_1
        # Source Nodes: [x_324, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf323 = extern_kernels.convolution(buf322, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg266_1
        del buf322
        buf324 = reinterpret_tensor(buf283, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf283  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_87.run(buf318, buf323, arg267_1, buf324, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg267_1
        del buf318
        del buf323
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325, x_326, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf325 = extern_kernels.convolution(buf324, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (8, 264, 7, 7), (12936, 49, 7, 1))
        del arg268_1
        del buf324
        buf326 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_88.run(buf325, arg399_1, arg400_1, arg109_1, arg110_1, buf326, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del arg109_1
        del arg110_1
        del arg399_1
        del arg400_1
        del buf325
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1584, 7, 7), (77616, 49, 7, 1))
        del arg269_1
        buf328 = buf327; del buf327  # reuse
        buf329 = buf328; del buf328  # reuse
        # Source Nodes: [x_332, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_89.run(buf329, arg401_1, arg402_1, arg111_1, arg112_1, 620928, grid=grid(620928), stream=stream0)
        del arg111_1
        del arg112_1
        del arg401_1
        del arg402_1
        buf330 = empty_strided((8, 396, 7, 7), (19404, 1, 2772, 396), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf329, buf330, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, arg270_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf331, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg270_1
        buf332 = buf330; del buf330  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf329, buf332, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, arg271_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf333, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg271_1
        buf334 = buf332; del buf332  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf329, buf334, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, arg272_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf335, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg272_1
        buf336 = buf334; del buf334  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf329, buf336, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, arg273_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf337, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg273_1
        del buf336
        buf338 = buf329; del buf329  # reuse
        buf339 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf340 = reinterpret_tensor(buf339, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf339  # reuse
        # Source Nodes: [cat_46, x_338, x_341, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_94.run(buf340, buf331, buf333, buf335, buf337, arg403_1, arg404_1, arg113_1, arg114_1, buf338, 12672, 49, grid=grid(12672), stream=stream0)
        del arg113_1
        del arg114_1
        del arg403_1
        del arg404_1
        del buf331
        del buf333
        del buf335
        # Source Nodes: [x_341, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf341 = extern_kernels.convolution(buf340, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg274_1
        del buf340
        buf342 = reinterpret_tensor(buf341, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf341  # reuse
        # Source Nodes: [x_341, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_95.run(buf342, arg275_1, 1056, grid=grid(1056), stream=stream0)
        del arg275_1
        # Source Nodes: [x_341, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf343 = extern_kernels.convolution(buf342, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg276_1
        del buf342
        buf344 = buf338; del buf338  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_341, x_342, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_96.run(buf344, buf343, arg277_1, 620928, grid=grid(620928), stream=stream0)
        del arg277_1
        buf345 = empty_strided((8, 792, 7, 7), (38808, 1, 5544, 792), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf344, buf345, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg278_1
        buf347 = buf345; del buf345  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf344, buf347, 6336, 49, grid=grid(6336, 49), stream=stream0)
        del buf344
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg279_1
        buf349 = buf326; del buf326  # reuse
        # Source Nodes: [cat_45, shortcut_17, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_99.run(buf349, buf346, buf348, arg405_1, arg406_1, arg115_1, arg116_1, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del arg115_1
        del arg116_1
        del arg405_1
        del arg406_1
        del buf346
        del buf348
        # Source Nodes: [x_350], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 1584, 7, 7), (77616, 49, 7, 1))
        del arg280_1
        buf351 = buf350; del buf350  # reuse
        buf352 = buf351; del buf351  # reuse
        # Source Nodes: [x_351, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_89.run(buf352, arg407_1, arg408_1, arg117_1, arg118_1, 620928, grid=grid(620928), stream=stream0)
        del arg117_1
        del arg118_1
        del arg407_1
        del arg408_1
        buf353 = reinterpret_tensor(buf337, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf337  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf352, buf353, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf354, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg281_1
        buf355 = buf353; del buf353  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf352, buf355, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, arg282_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf356, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg282_1
        buf357 = buf355; del buf355  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf352, buf357, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, arg283_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf358, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg283_1
        buf359 = buf357; del buf357  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf352, buf359, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, arg284_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf360, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg284_1
        del buf359
        buf361 = buf352; del buf352  # reuse
        buf362 = reinterpret_tensor(buf343, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf343  # reuse
        buf363 = reinterpret_tensor(buf362, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf362  # reuse
        # Source Nodes: [cat_44, x_357, x_360, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_94.run(buf363, buf354, buf356, buf358, buf360, arg409_1, arg410_1, arg119_1, arg120_1, buf361, 12672, 49, grid=grid(12672), stream=stream0)
        del arg119_1
        del arg120_1
        del arg409_1
        del arg410_1
        del buf354
        del buf356
        del buf358
        # Source Nodes: [x_360, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf364 = extern_kernels.convolution(buf363, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg285_1
        del buf363
        buf365 = reinterpret_tensor(buf364, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf364  # reuse
        # Source Nodes: [x_360, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_95.run(buf365, arg286_1, 1056, grid=grid(1056), stream=stream0)
        del arg286_1
        # Source Nodes: [x_360, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf366 = extern_kernels.convolution(buf365, arg287_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg287_1
        del buf365
        buf367 = buf361; del buf361  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_360, x_361, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_96.run(buf367, buf366, arg288_1, 620928, grid=grid(620928), stream=stream0)
        del arg288_1
        buf368 = buf347; del buf347  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf367, buf368, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg289_1
        buf370 = buf368; del buf368  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf367, buf370, 6336, 49, grid=grid(6336, 49), stream=stream0)
        del buf367
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg290_1
        buf372 = buf349; del buf349  # reuse
        # Source Nodes: [cat_43, shortcut_18, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_99.run(buf372, buf369, buf371, arg411_1, arg412_1, arg121_1, arg122_1, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del arg121_1
        del arg122_1
        del arg411_1
        del arg412_1
        del buf369
        del buf371
        # Source Nodes: [x_369], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 1584, 7, 7), (77616, 49, 7, 1))
        del arg291_1
        buf374 = buf373; del buf373  # reuse
        buf375 = buf374; del buf374  # reuse
        # Source Nodes: [x_370, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_89.run(buf375, arg413_1, arg414_1, arg123_1, arg124_1, 620928, grid=grid(620928), stream=stream0)
        del arg123_1
        del arg124_1
        del arg413_1
        del arg414_1
        buf376 = reinterpret_tensor(buf360, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf360  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf375, buf376, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, arg292_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf377, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg292_1
        buf378 = buf376; del buf376  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf375, buf378, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, arg293_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf379, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg293_1
        buf380 = buf378; del buf378  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf375, buf380, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, arg294_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf381, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg294_1
        buf382 = buf380; del buf380  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf375, buf382, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, arg295_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf383, (8, 396, 7, 7), (19404, 49, 7, 1))
        del arg295_1
        del buf382
        buf384 = buf375; del buf375  # reuse
        buf385 = reinterpret_tensor(buf366, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf366  # reuse
        buf386 = reinterpret_tensor(buf385, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf385  # reuse
        # Source Nodes: [cat_42, x_376, x_379, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_94.run(buf386, buf377, buf379, buf381, buf383, arg415_1, arg416_1, arg125_1, arg126_1, buf384, 12672, 49, grid=grid(12672), stream=stream0)
        del arg125_1
        del arg126_1
        del arg415_1
        del arg416_1
        del buf377
        del buf379
        del buf381
        del buf383
        # Source Nodes: [x_379, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf387 = extern_kernels.convolution(buf386, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg296_1
        del buf386
        buf388 = reinterpret_tensor(buf387, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf387  # reuse
        # Source Nodes: [x_379, x_se_60, x_se_61, x_se_62], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_95.run(buf388, arg297_1, 1056, grid=grid(1056), stream=stream0)
        del arg297_1
        # Source Nodes: [x_379, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf389 = extern_kernels.convolution(buf388, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg298_1
        del buf388
        buf390 = buf384; del buf384  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_379, x_380, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_96.run(buf390, buf389, arg299_1, 620928, grid=grid(620928), stream=stream0)
        del arg299_1
        del buf389
        buf391 = buf370; del buf370  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf390, buf391, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg300_1
        buf393 = buf391; del buf391  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf390, buf393, 6336, 49, grid=grid(6336, 49), stream=stream0)
        del buf390
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 132, 7, 7), (6468, 49, 7, 1))
        del arg301_1
        del buf393
        buf395 = buf372; del buf372  # reuse
        # Source Nodes: [cat_41, x_383, x_388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_99.run(buf395, buf392, buf394, arg417_1, arg418_1, arg127_1, arg128_1, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del arg127_1
        del arg128_1
        del arg417_1
        del arg418_1
        del buf392
        del buf394
        # Source Nodes: [cat_41, x_383, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
        buf396 = extern_kernels.convolution(buf395, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 1536, 7, 7), (75264, 49, 7, 1))
        del arg302_1
        del buf395
        buf397 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf398 = reinterpret_tensor(buf397, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf397  # reuse
        # Source Nodes: [x_390, x_394, x_395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_100.run(buf398, buf396, arg419_1, arg420_1, arg129_1, arg130_1, 12288, 49, grid=grid(12288), stream=stream0)
        del arg129_1
        del arg130_1
        del arg419_1
        del arg420_1
        del buf396
        buf399 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg304_1, reinterpret_tensor(buf398, (8, 1536), (1536, 1), 0), reinterpret_tensor(arg303_1, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf399)
        del arg303_1
        del arg304_1
        return (buf399, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1000, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
