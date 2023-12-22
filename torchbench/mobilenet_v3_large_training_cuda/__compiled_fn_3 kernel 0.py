
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtjh24prmexvvkuuwa353z2t4zcrhpxepjdtjmzrod7rpi6wxxd.py
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
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qt35hgivxfbpbseop5fngos3x46xcuadtsy67mhdutjtzlhsw4.py
# Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
# l__mod___features_0_0 => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmk52ziwvhyyxkihpy5xdqs3nrguv7rzypyyi7xypakidvnzr6eu.py
# Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# l__mod___features_0_1 => add_1, mul_1, mul_2, sub
# l__mod___features_0_2 => add_2, clamp_max, clamp_min, div, mul_3
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5ji5qvpns3zfoo22dqdfbpamrgkghsheftyqn5jk2lh3elip72.py
# Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___1___block_0_1 => add_4, mul_5, mul_6, sub_1
# getattr_l__mod___features___1___block_0_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iet464m6yrptjxnlbo5liwkatjunceezibgreshlerv7yazvuo.py
# Source Nodes: [result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result => add_6, mul_8, mul_9, sub_2
# result_1 => add_7
triton_poi_fused__native_batch_norm_legit_no_training_add_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxluz7akj4qmgpp7jyuo23yop6yvs2kgnr26nbl2s7p2xu5krrxi.py
# Source Nodes: [getattr_l__mod___features___2___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___block_0_0 => convolution_3
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzwcpv7ozqclguqeeb3rrtetzhxfozppkjxw7rhdnvpggrjas34.py
# Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___2___block_0_1 => add_9, mul_11, mul_12, sub_3
# getattr_l__mod___features___2___block_0_2 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsl55f6bi3n2zjc7p3qnmdyr2e2s4tnvx277w23ivhl4jgorq2o.py
# Source Nodes: [getattr_l__mod___features___2___block_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___block_1_0 => convolution_4
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cer7gzwnrrwsu6nixko4ao4xc3uuq3byaxg33s5clvogjzhqcp63.py
# Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___2___block_1_1 => add_11, mul_14, mul_15, sub_4
# getattr_l__mod___features___2___block_1_2 => relu_2
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
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3uhz76qk6cqihblerx6fxs5dpuzfsjees52yr7u2auccjprugb.py
# Source Nodes: [getattr_l__mod___features___2___block_2_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___block_2_0 => convolution_5
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
    ynumel = 96
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hab5i7np6nbui3bkemiprqfcpcitlar3oksvct45eztc4ffkgj.py
# Source Nodes: [result_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_2 => add_13, mul_17, mul_18, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
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
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tcjtecnttmym5idgp5shecorj4dguczvch4dnaqrumv2fdz67u.py
# Source Nodes: [getattr_l__mod___features___3___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___3___block_0_0 => convolution_6
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (225792*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65fxw4xbkgn4dee6kdahm7maakkivtc4zgv5dvd64tqxw6435wv.py
# Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___3___block_0_1 => add_15, mul_20, mul_21, sub_6
# getattr_l__mod___features___3___block_0_2 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kr3upmaccxdw27o6upfdyl4zw2fxtjce4prpxsjb3gray53qyq.py
# Source Nodes: [result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_3 => add_19, mul_26, mul_27, sub_8
# result_4 => add_20
triton_poi_fused__native_batch_norm_legit_no_training_add_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
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
    tmp15 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dw7telp4vujuxl75ztq37pplsqcxabhm765q2svjofmjnasppw.py
# Source Nodes: [getattr_l__mod___features___4___block_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___4___block_1_0 => convolution_10
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2yc5r2qmjzwkgqmku6gdeobmyzhyjl7xvo4ae3rqznke4yibka.py
# Source Nodes: [getattr_l__mod___features___4___block_1_1, getattr_l__mod___features___4___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___4___block_1_1 => add_24, mul_32, mul_33, sub_10
# getattr_l__mod___features___4___block_1_2 => relu_6
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnpsydrxbum576szx32xam4etpunyl7xrdslzeymz6xcsc43vuyi.py
# Source Nodes: [scale], Original ATen: [aten.mean]
# scale => mean
triton_red_fused_mean_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2016
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (8064*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtfbqsp6jxmuimowsxwj7lazcclvxvjxxjd3wq4xziivk5vzdg3.py
# Source Nodes: [scale], Original ATen: [aten.mean]
# scale => mean
triton_per_fused_mean_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 72
    x1 = (xindex // 72)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (504*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3v5naqzf67gmn2jmcimbpjdjtuouz3btqr2kdzz4axsxeutkyaf.py
# Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.relu]
# scale_1 => convolution_11
# scale_2 => relu_7
triton_poi_fused_convolution_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4yfoysxcamnb6rkrg2y75uqyunveuwqaa26bysiwszju2ogwuh.py
# Source Nodes: [scale_3, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid]
# scale_3 => convolution_12
# scale_4 => add_25, clamp_max_1, clamp_min_1, div_1
triton_poi_fused_convolution_hardsigmoid_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dxh6os334yjqh57yfcdhrth5vjxerg6s7tiysxmwltsaf5urly.py
# Source Nodes: [mul], Original ATen: [aten.mul]
# mul => mul_34
triton_poi_fused_mul_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 72
    x2 = (xindex // 56448)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqckhfiy2k7nqpktqipau5sqhn2ekhbv6ao4lmyizs27ciuvb5q.py
# Source Nodes: [getattr_l__mod___features___4___block_3_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___4___block_3_0 => convolution_13
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsqgwzr6l7m3wn74msjvkoxcoyymshps3iolbzzgb2fvemhtm7o.py
# Source Nodes: [result_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_5 => add_27, mul_36, mul_37, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmd3ng3dc7gzh7vbzljq3somtvivqhjtexyhnfqyzz6tyyjxtwsl.py
# Source Nodes: [getattr_l__mod___features___5___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___5___block_0_0 => convolution_14
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5p2rsx3afwfy3ebfafuauesnghqiklmzle6ivc3wezkv6ilocz.py
# Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___5___block_0_1 => add_29, mul_39, mul_40, sub_12
# getattr_l__mod___features___5___block_0_2 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybnbfmthi76gesd3udga4mzmoxhhwbogzwfzgo4ewgb3f6k2lwb.py
# Source Nodes: [scale_5], Original ATen: [aten.mean]
# scale_5 => mean_1
triton_red_fused_mean_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (13440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyiouxamfxsifd4sbzabmkxakdp47bdmsuwighmdb3ng4efuey7f.py
# Source Nodes: [scale_5], Original ATen: [aten.mean]
# scale_5 => mean_1
triton_per_fused_mean_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (840*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgvri42ht77sldqg4rho5s7quxqswu6kce2ts73ahi6gncjhfnp.py
# Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.relu]
# scale_6 => convolution_16
# scale_7 => relu_10
triton_poi_fused_convolution_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clkh56vhx2ivtlssxxyvng3pn6vj5xs7fjm3v4jxjn4jkoehk4qm.py
# Source Nodes: [scale_8, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid]
# scale_8 => convolution_17
# scale_9 => add_32, clamp_max_2, clamp_min_2, div_2
triton_poi_fused_convolution_hardsigmoid_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsupa2oqbh2oxwoiv3kpc3b5sjgpijzro2pvbsprsmaakrzjjl4.py
# Source Nodes: [mul_1], Original ATen: [aten.mul]
# mul_1 => mul_44
triton_poi_fused_mul_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 120
    x2 = (xindex // 94080)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzufsfcwx2hggszk7hrjo43xpotqazsquaw6yxadbyqwbmh4nsr.py
# Source Nodes: [result_6, result_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_6 => add_34, mul_46, mul_47, sub_14
# result_7 => add_35
triton_poi_fused__native_batch_norm_legit_no_training_add_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jw32pxmostoomfbmofgpm5rfrbvywkteh6fw7qsngfgyrgknkk.py
# Source Nodes: [getattr_l__mod___features___7___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___7___block_0_0 => convolution_24
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fck76wc6kejzhjvb2xdjzamvqhrjo7ixk7zy4misttzejerwmy.py
# Source Nodes: [getattr_l__mod___features___7___block_0_1, getattr_l__mod___features___7___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___7___block_0_1 => add_45, mul_59, mul_60, sub_18
# getattr_l__mod___features___7___block_0_2 => add_46, clamp_max_4, clamp_min_4, div_4, mul_61
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujjep74r6o3rkeeownso7y3asg5oaqz4rf6pemc5ip5xy73l2zx.py
# Source Nodes: [getattr_l__mod___features___7___block_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___7___block_1_0 => convolution_25
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdguj6pragq6hgia7d4dd4xzisjygkmmrwlvtzi2hiat6uzdnsji.py
# Source Nodes: [getattr_l__mod___features___7___block_1_1, getattr_l__mod___features___7___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___7___block_1_1 => add_48, mul_63, mul_64, sub_19
# getattr_l__mod___features___7___block_1_2 => add_49, clamp_max_5, clamp_min_5, div_5, mul_65
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7ntyovbwnxhftvpng2ssqvj43s4kzgg7irns2xov4fk2ol2pfx.py
# Source Nodes: [getattr_l__mod___features___7___block_2_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___7___block_2_0 => convolution_26
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2m3nswd2nblcue7ck7ylkuxe5slmwulnwz3ltfbjja7vcmbcrzn.py
# Source Nodes: [result_10], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_10 => add_51, mul_67, mul_68, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cveq6ipdyzgum7yv24r5xyz6lwwpkxdlvavshj3pdfptpphbh6tu.py
# Source Nodes: [getattr_l__mod___features___8___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___8___block_0_0 => convolution_27
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 800
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 200
    y1 = (yindex // 200)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (200*x2) + (39200*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czalqzuusfe3nb3y3pkrlomtaewa4vbprcqke3uhntwe3ely5cvz.py
# Source Nodes: [getattr_l__mod___features___8___block_0_1, getattr_l__mod___features___8___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___8___block_0_1 => add_53, mul_70, mul_71, sub_21
# getattr_l__mod___features___8___block_0_2 => add_54, clamp_max_6, clamp_min_6, div_6, mul_72
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwzhmd7uvdeo66ueoeorl3gxjayhlhv2udl2xlj3kxowjqr3aw6t.py
# Source Nodes: [result_11, result_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_11 => add_59, mul_78, mul_79, sub_23
# result_12 => add_60
triton_poi_fused__native_batch_norm_legit_no_training_add_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/cregnyvi7hctdfyukert4xeq7anzhsqvprkfcuxet4pq7auncton.py
# Source Nodes: [getattr_l__mod___features___9___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___9___block_0_0 => convolution_30
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 736
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (184*x2) + (36064*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56u6nddvdskjon3s2jalmd7cnslczyqn6eykrlqlj2bwl7kkrrg.py
# Source Nodes: [getattr_l__mod___features___9___block_0_1, getattr_l__mod___features___9___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___9___block_0_1 => add_62, mul_81, mul_82, sub_24
# getattr_l__mod___features___9___block_0_2 => add_63, clamp_max_8, clamp_min_8, div_8, mul_83
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 184
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5k5ajip2ogdsz7bi37k6sxa4qjrc6h6qdmlzhoqiah3tq6utdx.py
# Source Nodes: [getattr_l__mod___features___11___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___11___block_0_0 => convolution_36
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3djnmkiqmef6xyl7rcu4odc2kfare2pboehj7rracuk6aq4tj6q.py
# Source Nodes: [getattr_l__mod___features___11___block_0_1, getattr_l__mod___features___11___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___11___block_0_1 => add_80, mul_103, mul_104, sub_30
# getattr_l__mod___features___11___block_0_2 => add_81, clamp_max_12, clamp_min_12, div_12, mul_105
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigkdp63mklabs2pd37qlnifbaydk6tlu3vwpdppf5vvwghtzqfv.py
# Source Nodes: [scale_15], Original ATen: [aten.mean]
# scale_15 => mean_3
triton_red_fused_mean_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csb7cysqr2qjs5vnpavb64dcqncli65a6zerwrzl2ozotozz7437.py
# Source Nodes: [scale_15], Original ATen: [aten.mean]
# scale_15 => mean_3
triton_per_fused_mean_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (960*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyr2j6y7gfnpat5rgsipmntnm2kepkd5iq7nff3nas44mybm3dmy.py
# Source Nodes: [scale_16, scale_17], Original ATen: [aten.convolution, aten.relu]
# scale_16 => convolution_38
# scale_17 => relu_14
triton_poi_fused_convolution_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hz/chz3tcdkgtmwodw4x35vrbqmhaaolzp3pkrqn23lsfyx3iic3utl.py
# Source Nodes: [scale_18, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid]
# scale_18 => convolution_39
# scale_19 => add_85, clamp_max_14, clamp_min_14, div_14
triton_poi_fused_convolution_hardsigmoid_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhkrv4cvt4tiiepm62ikvv5evo4do6pl5fgt65gqaxgruy2kufj.py
# Source Nodes: [mul_3], Original ATen: [aten.mul]
# mul_3 => mul_110
triton_poi_fused_mul_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 480
    x2 = (xindex // 94080)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/crelagua7fnk5fcj2tjv5k5f2eat3usosiaexleao65gfaxcz6cf.py
# Source Nodes: [getattr_l__mod___features___11___block_3_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___11___block_3_0 => convolution_40
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6v2zzlpxxdia6ug5tqm7levlzcdbtttlnj4gzoj75uvjhhpc2i.py
# Source Nodes: [result_17], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_17 => add_87, mul_112, mul_113, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2endllmw3x5bty3tlatqmdr5hiceisqiupsnlugnpk2reddmrj.py
# Source Nodes: [getattr_l__mod___features___12___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___12___block_0_0 => convolution_41
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7rtjkb4sak5oaqp4humncyl4hot6zpe2pk3f3gi5fyuhdy6x2g.py
# Source Nodes: [getattr_l__mod___features___12___block_0_1, getattr_l__mod___features___12___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___12___block_0_1 => add_89, mul_115, mul_116, sub_33
# getattr_l__mod___features___12___block_0_2 => add_90, clamp_max_15, clamp_min_15, div_15, mul_117
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33qcylh5aqgjyerucdzckvbhjvf5leed3jbpsqjkxzavcg7dycd.py
# Source Nodes: [scale_20], Original ATen: [aten.mean]
# scale_20 => mean_4
triton_red_fused_mean_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cfff4p7gew3bfyqvctxdauwnzrl4eow4gmkq7dpoqqjd46g5l2sq.py
# Source Nodes: [scale_20], Original ATen: [aten.mean]
# scale_20 => mean_4
triton_per_fused_mean_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (1344*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigr5w7aqnsfgnzf6xalrixp2xevjelba3dh6qd676sjzogjhk6v.py
# Source Nodes: [scale_21, scale_22], Original ATen: [aten.convolution, aten.relu]
# scale_21 => convolution_43
# scale_22 => relu_15
triton_poi_fused_convolution_relu_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 168
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcbpmvjqntoq3y57h7ppbupbjuaxbwlbxyfmqp7q3wkr6k4klrq.py
# Source Nodes: [scale_23, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid]
# scale_23 => convolution_44
# scale_24 => add_94, clamp_max_17, clamp_min_17, div_17
triton_poi_fused_convolution_hardsigmoid_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqcucxx6yspqpht6svbmic6a7wrv27i3ex75qnk742jkytkkfz5.py
# Source Nodes: [mul_4], Original ATen: [aten.mul]
# mul_4 => mul_122
triton_poi_fused_mul_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 672
    x2 = (xindex // 131712)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w4yaab7ahujy46lk4x2mvsfzg4ioogjumijwg4zok67ccoib7z.py
# Source Nodes: [result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_18 => add_96, mul_124, mul_125, sub_35
# result_19 => add_97
triton_poi_fused__native_batch_norm_legit_no_training_add_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3wnj7fuxxi3s33fahkhcxw77begrmvlxvhjeoml5idvlrq25vf.py
# Source Nodes: [getattr_l__mod___features___13___block_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___13___block_1_0 => convolution_47
triton_poi_fused_convolution_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5lxrsg2nhzxq2v3vpadvrtusigfyxifbbg37yeh4iq6g6gi3jy.py
# Source Nodes: [getattr_l__mod___features___13___block_1_1, getattr_l__mod___features___13___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___13___block_1_1 => add_102, mul_131, mul_132, sub_37
# getattr_l__mod___features___13___block_1_2 => add_103, clamp_max_19, clamp_min_19, div_19, mul_133
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwm6lvd7xliln6y2smh2ey5hguvi3knh7d5mxwo3gx6c47xppcvk.py
# Source Nodes: [scale_25], Original ATen: [aten.mean]
# scale_25 => mean_5
triton_per_fused_mean_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/cht3pi5lyig2vegerl4phgrbzmfvplxhrommghotwyywilhkazcq.py
# Source Nodes: [scale_28, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# scale_28 => convolution_49
# scale_29 => add_104, clamp_max_20, clamp_min_20, div_20
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp10 = -3.0
    tmp11 = tmp2 > tmp10
    tmp12 = tmp2 < tmp3
    tmp13 = tmp11 & tmp12
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotnosld6j2vgmxtdzpghkxz2vspkob66mmj5xskrpiqcgj3mufn.py
# Source Nodes: [mul_5], Original ATen: [aten.mul]
# mul_5 => mul_134
triton_poi_fused_mul_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 672
    x2 = (xindex // 32928)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvt3f5c2f3fla4cnno3f7uhfhx45f2hiy5awd7tkpujjvowzllxu.py
# Source Nodes: [getattr_l__mod___features___13___block_3_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___13___block_3_0 => convolution_50
triton_poi_fused_convolution_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/cey4t3qvbf62hrkz4dvlfjganhexdcxxln6e7c74utkotach5agj.py
# Source Nodes: [result_20], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_20 => add_106, mul_136, mul_137, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chd25ol3tz3ni5txgchmzfqpum3su5jkke4aqosdte3n35scu6m6.py
# Source Nodes: [getattr_l__mod___features___14___block_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___14___block_0_0 => convolution_51
triton_poi_fused_convolution_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
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
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfkheu3mxj4pk5kiteechcwwoye2sgmckndl7bciniws56bbkhk.py
# Source Nodes: [getattr_l__mod___features___14___block_0_1, getattr_l__mod___features___14___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___14___block_0_1 => add_108, mul_139, mul_140, sub_39
# getattr_l__mod___features___14___block_0_2 => add_109, clamp_max_21, clamp_min_21, div_21, mul_141
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3d4jdbpahyviweiqe5qtglphcwfmle2e5ea3scnn2oihepttrf.py
# Source Nodes: [scale_30], Original ATen: [aten.mean]
# scale_30 => mean_6
triton_per_fused_mean_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_69', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rstrni7qtfxiqqxj4a7phyzqlyl7ogpmhsbi4s3d7hfmjpicop.py
# Source Nodes: [scale_31, scale_32], Original ATen: [aten.convolution, aten.relu]
# scale_31 => convolution_53
# scale_32 => relu_17
triton_poi_fused_convolution_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_70', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyz3r2frufs4hvu2myrpmjm2e4q6s6qqb5ntziiyftls2jqxgadp.py
# Source Nodes: [scale_33, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# scale_33 => convolution_54
# scale_34 => add_113, clamp_max_23, clamp_min_23, div_23
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp10 = -3.0
    tmp11 = tmp2 > tmp10
    tmp12 = tmp2 < tmp3
    tmp13 = tmp11 & tmp12
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu45zx5kymzxgadfaykbbpmnzmt4hoxlt2ira3nbpvq74cluf4zn.py
# Source Nodes: [mul_6], Original ATen: [aten.mul]
# mul_6 => mul_146
triton_poi_fused_mul_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 960
    x2 = (xindex // 47040)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2akskfqyyn27yky3cwrihetoajlhowfqrxxc3oixxlbzzehzxc3.py
# Source Nodes: [result_21, result_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_21 => add_115, mul_148, mul_149, sub_41
# result_22 => add_116
triton_poi_fused__native_batch_norm_legit_no_training_add_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxng3i7o64tazqcdzzodroj2o4hr66xgyalocca7hlsxbsbjham.py
# Source Nodes: [l__mod___features_16_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# l__mod___features_16_1 => add_128, mul_163, mul_164, sub_45
triton_poi_fused__native_batch_norm_legit_no_training_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gy4sgfrfxciohaatlxlcx44r5enaqtxl7psiwe62jcaoyn2naa.py
# Source Nodes: [x, x_1, x_2], Original ATen: [aten.hardswish, aten.mean, aten.view]
# x => add_129, clamp_max_27, clamp_min_27, div_27, mul_165
# x_1 => mean_8
# x_2 => view
triton_per_fused_hardswish_mean_view_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_view_75', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7urquukl6j74c5vjuvx5g4r3l2x7zueds7gwjoqxqv6pdhxnmt.py
# Source Nodes: [l__mod___classifier_1], Original ATen: [aten.hardswish]
# l__mod___classifier_1 => add_130, clamp_max_28, clamp_min_28, div_28, mul_166
triton_poi_fused_hardswish_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civomav7pgnby6pcrw46xoeohtfeiuz2lx3u56qyknx5qnkhceyg.py
# Source Nodes: [scale_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# scale_23 => convolution_44
triton_poi_fused_convolution_hardsigmoid_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzj5lfsgvvgjx2bdiowyu2d2dmi4hqar6djb52r7juzbhagernz.py
# Source Nodes: [scale_18], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# scale_18 => convolution_39
triton_poi_fused_convolution_hardsigmoid_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbecytvoacjc4rqvkit6cfvoujsem6wdxbmvohnsvmf53jt7ail.py
# Source Nodes: [scale_13], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# scale_13 => convolution_22
triton_poi_fused_convolution_hardsigmoid_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/cze2twzs37geb3nt2dhsn2r65v5k4mrprnenqvyflg3jjeprr4ll.py
# Source Nodes: [scale_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# scale_3 => convolution_12
triton_poi_fused_convolution_hardsigmoid_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_24, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_30, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_33, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_37, (72, ), (1, ))
    assert_size_stride(primals_38, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_40, (40, ), (1, ))
    assert_size_stride(primals_41, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_42, (120, ), (1, ))
    assert_size_stride(primals_43, (120, ), (1, ))
    assert_size_stride(primals_44, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_45, (120, ), (1, ))
    assert_size_stride(primals_46, (120, ), (1, ))
    assert_size_stride(primals_47, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_51, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_52, (40, ), (1, ))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_54, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_57, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_58, (120, ), (1, ))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_60, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_63, (120, ), (1, ))
    assert_size_stride(primals_64, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_65, (40, ), (1, ))
    assert_size_stride(primals_66, (40, ), (1, ))
    assert_size_stride(primals_67, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_69, (240, ), (1, ))
    assert_size_stride(primals_70, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (240, ), (1, ))
    assert_size_stride(primals_72, (240, ), (1, ))
    assert_size_stride(primals_73, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_74, (80, ), (1, ))
    assert_size_stride(primals_75, (80, ), (1, ))
    assert_size_stride(primals_76, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_77, (200, ), (1, ))
    assert_size_stride(primals_78, (200, ), (1, ))
    assert_size_stride(primals_79, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (200, ), (1, ))
    assert_size_stride(primals_81, (200, ), (1, ))
    assert_size_stride(primals_82, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_83, (80, ), (1, ))
    assert_size_stride(primals_84, (80, ), (1, ))
    assert_size_stride(primals_85, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_86, (184, ), (1, ))
    assert_size_stride(primals_87, (184, ), (1, ))
    assert_size_stride(primals_88, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_89, (184, ), (1, ))
    assert_size_stride(primals_90, (184, ), (1, ))
    assert_size_stride(primals_91, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_92, (80, ), (1, ))
    assert_size_stride(primals_93, (80, ), (1, ))
    assert_size_stride(primals_94, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_95, (184, ), (1, ))
    assert_size_stride(primals_96, (184, ), (1, ))
    assert_size_stride(primals_97, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (184, ), (1, ))
    assert_size_stride(primals_99, (184, ), (1, ))
    assert_size_stride(primals_100, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_101, (80, ), (1, ))
    assert_size_stride(primals_102, (80, ), (1, ))
    assert_size_stride(primals_103, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_104, (480, ), (1, ))
    assert_size_stride(primals_105, (480, ), (1, ))
    assert_size_stride(primals_106, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (480, ), (1, ))
    assert_size_stride(primals_108, (480, ), (1, ))
    assert_size_stride(primals_109, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_110, (120, ), (1, ))
    assert_size_stride(primals_111, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_112, (480, ), (1, ))
    assert_size_stride(primals_113, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_114, (112, ), (1, ))
    assert_size_stride(primals_115, (112, ), (1, ))
    assert_size_stride(primals_116, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_117, (672, ), (1, ))
    assert_size_stride(primals_118, (672, ), (1, ))
    assert_size_stride(primals_119, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_120, (672, ), (1, ))
    assert_size_stride(primals_121, (672, ), (1, ))
    assert_size_stride(primals_122, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_123, (168, ), (1, ))
    assert_size_stride(primals_124, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_125, (672, ), (1, ))
    assert_size_stride(primals_126, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_127, (112, ), (1, ))
    assert_size_stride(primals_128, (112, ), (1, ))
    assert_size_stride(primals_129, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_130, (672, ), (1, ))
    assert_size_stride(primals_131, (672, ), (1, ))
    assert_size_stride(primals_132, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_133, (672, ), (1, ))
    assert_size_stride(primals_134, (672, ), (1, ))
    assert_size_stride(primals_135, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_136, (168, ), (1, ))
    assert_size_stride(primals_137, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_138, (672, ), (1, ))
    assert_size_stride(primals_139, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (960, ), (1, ))
    assert_size_stride(primals_144, (960, ), (1, ))
    assert_size_stride(primals_145, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (960, ), (1, ))
    assert_size_stride(primals_147, (960, ), (1, ))
    assert_size_stride(primals_148, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_149, (240, ), (1, ))
    assert_size_stride(primals_150, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_151, (960, ), (1, ))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (160, ), (1, ))
    assert_size_stride(primals_154, (160, ), (1, ))
    assert_size_stride(primals_155, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_156, (960, ), (1, ))
    assert_size_stride(primals_157, (960, ), (1, ))
    assert_size_stride(primals_158, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_159, (960, ), (1, ))
    assert_size_stride(primals_160, (960, ), (1, ))
    assert_size_stride(primals_161, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_162, (240, ), (1, ))
    assert_size_stride(primals_163, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_164, (960, ), (1, ))
    assert_size_stride(primals_165, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_166, (160, ), (1, ))
    assert_size_stride(primals_167, (160, ), (1, ))
    assert_size_stride(primals_168, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_169, (960, ), (1, ))
    assert_size_stride(primals_170, (960, ), (1, ))
    assert_size_stride(primals_171, (1280, 960), (960, 1))
    assert_size_stride(primals_172, (1280, ), (1, ))
    assert_size_stride(primals_173, (1000, 1280), (1280, 1))
    assert_size_stride(primals_174, (1000, ), (1, ))
    assert_size_stride(primals_175, (16, ), (1, ))
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (24, ), (1, ))
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (72, ), (1, ))
    assert_size_stride(primals_194, (72, ), (1, ))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (24, ), (1, ))
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (72, ), (1, ))
    assert_size_stride(primals_203, (72, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (72, ), (1, ))
    assert_size_stride(primals_206, (72, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (40, ), (1, ))
    assert_size_stride(primals_209, (40, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (120, ), (1, ))
    assert_size_stride(primals_212, (120, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (40, ), (1, ))
    assert_size_stride(primals_218, (40, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (120, ), (1, ))
    assert_size_stride(primals_221, (120, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (120, ), (1, ))
    assert_size_stride(primals_224, (120, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (40, ), (1, ))
    assert_size_stride(primals_227, (40, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (240, ), (1, ))
    assert_size_stride(primals_230, (240, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (240, ), (1, ))
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (80, ), (1, ))
    assert_size_stride(primals_236, (80, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (200, ), (1, ))
    assert_size_stride(primals_239, (200, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (200, ), (1, ))
    assert_size_stride(primals_242, (200, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (80, ), (1, ))
    assert_size_stride(primals_245, (80, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (184, ), (1, ))
    assert_size_stride(primals_248, (184, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (184, ), (1, ))
    assert_size_stride(primals_251, (184, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (80, ), (1, ))
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (184, ), (1, ))
    assert_size_stride(primals_257, (184, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (184, ), (1, ))
    assert_size_stride(primals_260, (184, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (80, ), (1, ))
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (480, ), (1, ))
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (112, ), (1, ))
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (672, ), (1, ))
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (672, ), (1, ))
    assert_size_stride(primals_278, (672, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (112, ), (1, ))
    assert_size_stride(primals_281, (112, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (672, ), (1, ))
    assert_size_stride(primals_284, (672, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (672, ), (1, ))
    assert_size_stride(primals_287, (672, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (160, ), (1, ))
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (960, ), (1, ))
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (160, ), (1, ))
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (960, ), (1, ))
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (960, ), (1, ))
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_313, buf1, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_313
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf3 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf4 = reinterpret_tensor(buf2, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        buf5 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3.run(buf3, primals_175, primals_176, primals_2, primals_3, buf4, buf5, 802816, grid=grid(802816), stream=stream0)
        del primals_3
        # Source Nodes: [getattr_l__mod___features___1___block_0_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf6, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf7 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___1___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf6, buf7, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf6  # reuse
        # Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf7, primals_178, primals_179, primals_5, primals_6, buf8, 802816, grid=grid(802816), stream=stream0)
        del primals_6
        # Source Nodes: [getattr_l__mod___features___1___block_1_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf10 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___1___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf9, buf10, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf9  # reuse
        # Source Nodes: [result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_5.run(buf10, primals_181, primals_182, primals_8, primals_9, buf5, buf11, 802816, grid=grid(802816), stream=stream0)
        del primals_9
        # Source Nodes: [getattr_l__mod___features___2___block_0_0], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 112, 112), (802816, 12544, 112, 1))
        buf13 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf12, buf13, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (4, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf12  # reuse
        # Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf13, primals_184, primals_185, primals_11, primals_12, buf14, 3211264, grid=grid(3211264), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_l__mod___features___2___block_1_0], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf15, (4, 64, 56, 56), (200704, 3136, 56, 1))
        buf16 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf15, buf16, 256, 3136, grid=grid(256, 3136), stream=stream0)
        buf17 = reinterpret_tensor(buf15, (4, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf15  # reuse
        # Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf16, primals_187, primals_188, primals_14, primals_15, buf17, 802816, grid=grid(802816), stream=stream0)
        del primals_15
        # Source Nodes: [getattr_l__mod___features___2___block_2_0], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf19 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf18, buf19, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf18  # reuse
        # Source Nodes: [result_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf19, primals_190, primals_191, primals_17, primals_18, buf20, 301056, grid=grid(301056), stream=stream0)
        del primals_18
        # Source Nodes: [getattr_l__mod___features___3___block_0_0], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf22 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf21, buf22, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf21  # reuse
        # Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf22, primals_193, primals_194, primals_20, primals_21, buf23, 903168, grid=grid(903168), stream=stream0)
        del primals_21
        # Source Nodes: [getattr_l__mod___features___3___block_1_0], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf24, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf25 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf24, buf25, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf26 = reinterpret_tensor(buf24, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf24  # reuse
        # Source Nodes: [getattr_l__mod___features___3___block_1_1, getattr_l__mod___features___3___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf25, primals_196, primals_197, primals_23, primals_24, buf26, 903168, grid=grid(903168), stream=stream0)
        del primals_24
        # Source Nodes: [getattr_l__mod___features___3___block_2_0], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf28 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf27, buf28, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf29 = reinterpret_tensor(buf27, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf27  # reuse
        # Source Nodes: [result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf28, primals_199, primals_200, primals_26, primals_27, buf20, buf29, 301056, grid=grid(301056), stream=stream0)
        del primals_27
        # Source Nodes: [getattr_l__mod___features___4___block_0_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf31 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf30, buf31, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf32 = reinterpret_tensor(buf30, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf30  # reuse
        # Source Nodes: [getattr_l__mod___features___4___block_0_1, getattr_l__mod___features___4___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf31, primals_202, primals_203, primals_29, primals_30, buf32, 903168, grid=grid(903168), stream=stream0)
        del primals_30
        # Source Nodes: [getattr_l__mod___features___4___block_1_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_31, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf33, (4, 72, 28, 28), (56448, 784, 28, 1))
        buf34 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf33, buf34, 288, 784, grid=grid(288, 784), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (4, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf33  # reuse
        # Source Nodes: [getattr_l__mod___features___4___block_1_1, getattr_l__mod___features___4___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf34, primals_205, primals_206, primals_32, primals_33, buf35, 225792, grid=grid(225792), stream=stream0)
        del primals_33
        buf36 = empty_strided((4, 72, 1, 1, 7), (504, 1, 2016, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.mean]
        triton_red_fused_mean_17.run(buf35, buf36, 2016, 112, grid=grid(2016), stream=stream0)
        buf37 = empty_strided((4, 72, 1, 1), (72, 1, 288, 288), device='cuda', dtype=torch.float32)
        buf38 = reinterpret_tensor(buf37, (4, 72, 1, 1), (72, 1, 72, 72), 0); del buf37  # reuse
        # Source Nodes: [scale], Original ATen: [aten.mean]
        triton_per_fused_mean_18.run(buf38, buf36, 288, 7, grid=grid(288), stream=stream0)
        del buf36
        # Source Nodes: [scale_1], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 24, 1, 1), (24, 1, 1, 1))
        buf40 = reinterpret_tensor(buf39, (4, 24, 1, 1), (24, 1, 24, 24), 0); del buf39  # reuse
        # Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_19.run(buf40, primals_35, 96, grid=grid(96), stream=stream0)
        del primals_35
        # Source Nodes: [scale_3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 72, 1, 1), (72, 1, 1, 1))
        buf42 = empty_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_3, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_20.run(buf41, primals_37, buf42, 288, grid=grid(288), stream=stream0)
        buf43 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul], Original ATen: [aten.mul]
        triton_poi_fused_mul_21.run(buf42, buf35, buf43, 225792, grid=grid(225792), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___4___block_3_0], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf45 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf44, buf45, 160, 784, grid=grid(160, 784), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf44  # reuse
        # Source Nodes: [result_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf45, primals_208, primals_209, primals_39, primals_40, buf46, 125440, grid=grid(125440), stream=stream0)
        del primals_40
        # Source Nodes: [getattr_l__mod___features___5___block_0_0], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf48 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf47, buf48, 480, 784, grid=grid(480, 784), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf47  # reuse
        # Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf48, primals_211, primals_212, primals_42, primals_43, buf49, 376320, grid=grid(376320), stream=stream0)
        del primals_43
        # Source Nodes: [getattr_l__mod___features___5___block_1_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_44, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf50, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf51 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf50, buf51, 480, 784, grid=grid(480, 784), stream=stream0)
        buf52 = reinterpret_tensor(buf50, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf50  # reuse
        # Source Nodes: [getattr_l__mod___features___5___block_1_1, getattr_l__mod___features___5___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf51, primals_214, primals_215, primals_45, primals_46, buf52, 376320, grid=grid(376320), stream=stream0)
        del primals_46
        buf53 = empty_strided((4, 120, 1, 1, 7), (840, 1, 3360, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_5], Original ATen: [aten.mean]
        triton_red_fused_mean_26.run(buf52, buf53, 3360, 112, grid=grid(3360), stream=stream0)
        buf54 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf55 = reinterpret_tensor(buf54, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf54  # reuse
        # Source Nodes: [scale_5], Original ATen: [aten.mean]
        triton_per_fused_mean_27.run(buf55, buf53, 480, 7, grid=grid(480), stream=stream0)
        # Source Nodes: [scale_6], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 1, 1), (32, 1, 1, 1))
        buf57 = reinterpret_tensor(buf56, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf56  # reuse
        # Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_28.run(buf57, primals_48, 128, grid=grid(128), stream=stream0)
        del primals_48
        # Source Nodes: [scale_8], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 120, 1, 1), (120, 1, 1, 1))
        buf59 = empty_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_8, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_29.run(buf58, primals_50, buf59, 480, grid=grid(480), stream=stream0)
        buf60 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_1], Original ATen: [aten.mul]
        triton_poi_fused_mul_30.run(buf59, buf52, buf60, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___5___block_3_0], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf62 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf61, buf62, 160, 784, grid=grid(160, 784), stream=stream0)
        buf63 = reinterpret_tensor(buf61, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf61  # reuse
        # Source Nodes: [result_6, result_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_31.run(buf62, primals_217, primals_218, primals_52, primals_53, buf46, buf63, 125440, grid=grid(125440), stream=stream0)
        del primals_53
        # Source Nodes: [getattr_l__mod___features___6___block_0_0], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf65 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf64, buf65, 480, 784, grid=grid(480, 784), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf64  # reuse
        # Source Nodes: [getattr_l__mod___features___6___block_0_1, getattr_l__mod___features___6___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf65, primals_220, primals_221, primals_55, primals_56, buf66, 376320, grid=grid(376320), stream=stream0)
        del primals_56
        # Source Nodes: [getattr_l__mod___features___6___block_1_0], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_57, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf67, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf68 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf67, buf68, 480, 784, grid=grid(480, 784), stream=stream0)
        buf69 = reinterpret_tensor(buf67, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf67  # reuse
        # Source Nodes: [getattr_l__mod___features___6___block_1_1, getattr_l__mod___features___6___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf68, primals_223, primals_224, primals_58, primals_59, buf69, 376320, grid=grid(376320), stream=stream0)
        del primals_59
        buf70 = buf53; del buf53  # reuse
        # Source Nodes: [scale_10], Original ATen: [aten.mean]
        triton_red_fused_mean_26.run(buf69, buf70, 3360, 112, grid=grid(3360), stream=stream0)
        buf71 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf72 = reinterpret_tensor(buf71, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf71  # reuse
        # Source Nodes: [scale_10], Original ATen: [aten.mean]
        triton_per_fused_mean_27.run(buf72, buf70, 480, 7, grid=grid(480), stream=stream0)
        del buf70
        # Source Nodes: [scale_11], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 32, 1, 1), (32, 1, 1, 1))
        buf74 = reinterpret_tensor(buf73, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf73  # reuse
        # Source Nodes: [scale_11, scale_12], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_28.run(buf74, primals_61, 128, grid=grid(128), stream=stream0)
        del primals_61
        # Source Nodes: [scale_13], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 120, 1, 1), (120, 1, 1, 1))
        buf76 = empty_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_13, scale_14], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_29.run(buf75, primals_63, buf76, 480, grid=grid(480), stream=stream0)
        buf77 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_2], Original ATen: [aten.mul]
        triton_poi_fused_mul_30.run(buf76, buf69, buf77, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___6___block_3_0], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf79 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf78, buf79, 160, 784, grid=grid(160, 784), stream=stream0)
        buf80 = reinterpret_tensor(buf78, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf78  # reuse
        # Source Nodes: [result_8, result_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_31.run(buf79, primals_226, primals_227, primals_65, primals_66, buf63, buf80, 125440, grid=grid(125440), stream=stream0)
        del primals_66
        # Source Nodes: [getattr_l__mod___features___7___block_0_0], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 240, 28, 28), (188160, 784, 28, 1))
        buf82 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf81, buf82, 960, 784, grid=grid(960, 784), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (4, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf81  # reuse
        buf84 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_0_1, getattr_l__mod___features___7___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf82, primals_229, primals_230, primals_68, primals_69, buf83, buf84, 752640, grid=grid(752640), stream=stream0)
        del primals_69
        # Source Nodes: [getattr_l__mod___features___7___block_1_0], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf85, (4, 240, 14, 14), (47040, 196, 14, 1))
        buf86 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf85, buf86, 960, 196, grid=grid(960, 196), stream=stream0)
        buf87 = reinterpret_tensor(buf85, (4, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf85  # reuse
        buf88 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_1_1, getattr_l__mod___features___7___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35.run(buf86, primals_232, primals_233, primals_71, primals_72, buf87, buf88, 188160, grid=grid(188160), stream=stream0)
        del primals_72
        # Source Nodes: [getattr_l__mod___features___7___block_2_0], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf90 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf89, buf90, 320, 196, grid=grid(320, 196), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf89  # reuse
        # Source Nodes: [result_10], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf90, primals_235, primals_236, primals_74, primals_75, buf91, 62720, grid=grid(62720), stream=stream0)
        del primals_75
        # Source Nodes: [getattr_l__mod___features___8___block_0_0], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 200, 14, 14), (39200, 196, 14, 1))
        buf93 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf92, buf93, 800, 196, grid=grid(800, 196), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (4, 200, 14, 14), (39200, 1, 2800, 200), 0); del buf92  # reuse
        buf95 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_0_1, getattr_l__mod___features___8___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_39.run(buf93, primals_238, primals_239, primals_77, primals_78, buf94, buf95, 156800, grid=grid(156800), stream=stream0)
        del primals_78
        # Source Nodes: [getattr_l__mod___features___8___block_1_0], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf96, (4, 200, 14, 14), (39200, 196, 14, 1))
        buf97 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf96, buf97, 800, 196, grid=grid(800, 196), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (4, 200, 14, 14), (39200, 1, 2800, 200), 0); del buf96  # reuse
        buf99 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_1_1, getattr_l__mod___features___8___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_39.run(buf97, primals_241, primals_242, primals_80, primals_81, buf98, buf99, 156800, grid=grid(156800), stream=stream0)
        del primals_81
        # Source Nodes: [getattr_l__mod___features___8___block_2_0], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf101 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf100, buf101, 320, 196, grid=grid(320, 196), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf100  # reuse
        # Source Nodes: [result_11, result_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf101, primals_244, primals_245, primals_83, primals_84, buf91, buf102, 62720, grid=grid(62720), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_l__mod___features___9___block_0_0], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 184, 14, 14), (36064, 196, 14, 1))
        buf104 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf103, buf104, 736, 196, grid=grid(736, 196), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (4, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf103  # reuse
        buf106 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_0_1, getattr_l__mod___features___9___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42.run(buf104, primals_247, primals_248, primals_86, primals_87, buf105, buf106, 144256, grid=grid(144256), stream=stream0)
        del primals_87
        # Source Nodes: [getattr_l__mod___features___9___block_1_0], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf107, (4, 184, 14, 14), (36064, 196, 14, 1))
        buf108 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf107, buf108, 736, 196, grid=grid(736, 196), stream=stream0)
        buf109 = reinterpret_tensor(buf107, (4, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf107  # reuse
        buf110 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_1_1, getattr_l__mod___features___9___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42.run(buf108, primals_250, primals_251, primals_89, primals_90, buf109, buf110, 144256, grid=grid(144256), stream=stream0)
        del primals_90
        # Source Nodes: [getattr_l__mod___features___9___block_2_0], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf112 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf111, buf112, 320, 196, grid=grid(320, 196), stream=stream0)
        buf113 = reinterpret_tensor(buf111, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf111  # reuse
        # Source Nodes: [result_13, result_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf112, primals_253, primals_254, primals_92, primals_93, buf102, buf113, 62720, grid=grid(62720), stream=stream0)
        del primals_93
        # Source Nodes: [getattr_l__mod___features___10___block_0_0], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 184, 14, 14), (36064, 196, 14, 1))
        buf115 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf114, buf115, 736, 196, grid=grid(736, 196), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (4, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf114  # reuse
        buf117 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___block_0_1, getattr_l__mod___features___10___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42.run(buf115, primals_256, primals_257, primals_95, primals_96, buf116, buf117, 144256, grid=grid(144256), stream=stream0)
        del primals_96
        # Source Nodes: [getattr_l__mod___features___10___block_1_0], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf118, (4, 184, 14, 14), (36064, 196, 14, 1))
        buf119 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf118, buf119, 736, 196, grid=grid(736, 196), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (4, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf118  # reuse
        buf121 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___block_1_1, getattr_l__mod___features___10___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_42.run(buf119, primals_259, primals_260, primals_98, primals_99, buf120, buf121, 144256, grid=grid(144256), stream=stream0)
        del primals_99
        # Source Nodes: [getattr_l__mod___features___10___block_2_0], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf123 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___block_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf122, buf123, 320, 196, grid=grid(320, 196), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf122  # reuse
        # Source Nodes: [result_15, result_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf123, primals_262, primals_263, primals_101, primals_102, buf113, buf124, 62720, grid=grid(62720), stream=stream0)
        del primals_102
        # Source Nodes: [getattr_l__mod___features___11___block_0_0], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf126 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf125, buf126, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf125  # reuse
        buf128 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___block_0_1, getattr_l__mod___features___11___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44.run(buf126, primals_265, primals_266, primals_104, primals_105, buf127, buf128, 376320, grid=grid(376320), stream=stream0)
        del primals_105
        # Source Nodes: [getattr_l__mod___features___11___block_1_0], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf129, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf130 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf129, buf130, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf131 = reinterpret_tensor(buf129, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf129  # reuse
        buf132 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___block_1_1, getattr_l__mod___features___11___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44.run(buf130, primals_268, primals_269, primals_107, primals_108, buf131, buf132, 376320, grid=grid(376320), stream=stream0)
        del primals_108
        buf133 = empty_strided((4, 480, 1, 1, 2), (960, 1, 3840, 3840, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_15], Original ATen: [aten.mean]
        triton_red_fused_mean_45.run(buf132, buf133, 3840, 98, grid=grid(3840), stream=stream0)
        buf134 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf135 = reinterpret_tensor(buf134, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf134  # reuse
        # Source Nodes: [scale_15], Original ATen: [aten.mean]
        triton_per_fused_mean_46.run(buf135, buf133, 1920, 2, grid=grid(1920), stream=stream0)
        # Source Nodes: [scale_16], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 120, 1, 1), (120, 1, 1, 1))
        buf137 = reinterpret_tensor(buf136, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf136  # reuse
        # Source Nodes: [scale_16, scale_17], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_47.run(buf137, primals_110, 480, grid=grid(480), stream=stream0)
        del primals_110
        # Source Nodes: [scale_18], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 480, 1, 1), (480, 1, 1, 1))
        buf139 = empty_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_18, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_48.run(buf138, primals_112, buf139, 1920, grid=grid(1920), stream=stream0)
        buf140 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_3], Original ATen: [aten.mul]
        triton_poi_fused_mul_49.run(buf139, buf132, buf140, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___11___block_3_0], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 112, 14, 14), (21952, 196, 14, 1))
        buf142 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf141, buf142, 448, 196, grid=grid(448, 196), stream=stream0)
        buf143 = reinterpret_tensor(buf141, (4, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf141  # reuse
        # Source Nodes: [result_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf142, primals_271, primals_272, primals_114, primals_115, buf143, 87808, grid=grid(87808), stream=stream0)
        del primals_115
        # Source Nodes: [getattr_l__mod___features___12___block_0_0], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf145 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf144, buf145, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf144  # reuse
        buf147 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_0_1, getattr_l__mod___features___12___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_53.run(buf145, primals_274, primals_275, primals_117, primals_118, buf146, buf147, 526848, grid=grid(526848), stream=stream0)
        del primals_118
        # Source Nodes: [getattr_l__mod___features___12___block_1_0], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_119, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf148, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf149 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf148, buf149, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf150 = reinterpret_tensor(buf148, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf148  # reuse
        buf151 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_1_1, getattr_l__mod___features___12___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_53.run(buf149, primals_277, primals_278, primals_120, primals_121, buf150, buf151, 526848, grid=grid(526848), stream=stream0)
        del primals_121
        buf152 = empty_strided((4, 672, 1, 1, 2), (1344, 1, 5376, 5376, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_20], Original ATen: [aten.mean]
        triton_red_fused_mean_54.run(buf151, buf152, 5376, 98, grid=grid(5376), stream=stream0)
        buf153 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf154 = reinterpret_tensor(buf153, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf153  # reuse
        # Source Nodes: [scale_20], Original ATen: [aten.mean]
        triton_per_fused_mean_55.run(buf154, buf152, 2688, 2, grid=grid(2688), stream=stream0)
        del buf152
        # Source Nodes: [scale_21], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 168, 1, 1), (168, 1, 1, 1))
        buf156 = reinterpret_tensor(buf155, (4, 168, 1, 1), (168, 1, 168, 168), 0); del buf155  # reuse
        # Source Nodes: [scale_21, scale_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_56.run(buf156, primals_123, 672, grid=grid(672), stream=stream0)
        del primals_123
        # Source Nodes: [scale_23], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 672, 1, 1), (672, 1, 1, 1))
        buf158 = empty_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale_23, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_57.run(buf157, primals_125, buf158, 2688, grid=grid(2688), stream=stream0)
        buf159 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_4], Original ATen: [aten.mul]
        triton_poi_fused_mul_58.run(buf158, buf151, buf159, 526848, grid=grid(526848), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___12___block_3_0], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 112, 14, 14), (21952, 196, 14, 1))
        buf161 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf160, buf161, 448, 196, grid=grid(448, 196), stream=stream0)
        buf162 = reinterpret_tensor(buf160, (4, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf160  # reuse
        # Source Nodes: [result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_59.run(buf161, primals_280, primals_281, primals_127, primals_128, buf143, buf162, 87808, grid=grid(87808), stream=stream0)
        del primals_128
        # Source Nodes: [getattr_l__mod___features___13___block_0_0], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf164 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf163, buf164, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf165 = reinterpret_tensor(buf163, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf163  # reuse
        buf166 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_0_1, getattr_l__mod___features___13___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_53.run(buf164, primals_283, primals_284, primals_130, primals_131, buf165, buf166, 526848, grid=grid(526848), stream=stream0)
        del primals_131
        # Source Nodes: [getattr_l__mod___features___13___block_1_0], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_132, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf167, (4, 672, 7, 7), (32928, 49, 7, 1))
        buf168 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf167, buf168, 2688, 49, grid=grid(2688, 49), stream=stream0)
        buf169 = reinterpret_tensor(buf167, (4, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf167  # reuse
        buf170 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_1_1, getattr_l__mod___features___13___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_61.run(buf168, primals_286, primals_287, primals_133, primals_134, buf169, buf170, 131712, grid=grid(131712), stream=stream0)
        del primals_134
        buf171 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf172 = reinterpret_tensor(buf171, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf171  # reuse
        # Source Nodes: [scale_25], Original ATen: [aten.mean]
        triton_per_fused_mean_62.run(buf172, buf170, 2688, 49, grid=grid(2688), stream=stream0)
        # Source Nodes: [scale_26], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 168, 1, 1), (168, 1, 1, 1))
        buf174 = reinterpret_tensor(buf173, (4, 168, 1, 1), (168, 1, 168, 168), 0); del buf173  # reuse
        # Source Nodes: [scale_26, scale_27], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_56.run(buf174, primals_136, 672, grid=grid(672), stream=stream0)
        del primals_136
        # Source Nodes: [scale_28], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 672, 1, 1), (672, 1, 1, 1))
        buf176 = empty_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf227 = empty_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_28, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_63.run(buf175, primals_138, buf176, buf227, 2688, grid=grid(2688), stream=stream0)
        del buf175
        del primals_138
        buf177 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_5], Original ATen: [aten.mul]
        triton_poi_fused_mul_64.run(buf176, buf170, buf177, 131712, grid=grid(131712), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___13___block_3_0], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf179 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf178, buf179, 640, 49, grid=grid(640, 49), stream=stream0)
        buf180 = reinterpret_tensor(buf178, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf178  # reuse
        # Source Nodes: [result_20], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_66.run(buf179, primals_289, primals_290, primals_140, primals_141, buf180, 31360, grid=grid(31360), stream=stream0)
        del primals_141
        # Source Nodes: [getattr_l__mod___features___14___block_0_0], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf182 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf181, buf182, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf181  # reuse
        buf184 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___block_0_1, getattr_l__mod___features___14___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68.run(buf182, primals_292, primals_293, primals_143, primals_144, buf183, buf184, 188160, grid=grid(188160), stream=stream0)
        del primals_144
        # Source Nodes: [getattr_l__mod___features___14___block_1_0], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_145, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf185, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf186 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf185, buf186, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf185  # reuse
        buf188 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___block_1_1, getattr_l__mod___features___14___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68.run(buf186, primals_295, primals_296, primals_146, primals_147, buf187, buf188, 188160, grid=grid(188160), stream=stream0)
        del primals_147
        buf189 = reinterpret_tensor(buf133, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf133  # reuse
        buf190 = reinterpret_tensor(buf189, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf189  # reuse
        # Source Nodes: [scale_30], Original ATen: [aten.mean]
        triton_per_fused_mean_69.run(buf190, buf188, 3840, 49, grid=grid(3840), stream=stream0)
        # Source Nodes: [scale_31], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 240, 1, 1), (240, 1, 1, 1))
        buf192 = reinterpret_tensor(buf191, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf191  # reuse
        # Source Nodes: [scale_31, scale_32], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_70.run(buf192, primals_149, 960, grid=grid(960), stream=stream0)
        del primals_149
        # Source Nodes: [scale_33], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 960, 1, 1), (960, 1, 1, 1))
        buf194 = empty_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf226 = empty_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_33, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_71.run(buf193, primals_151, buf194, buf226, 3840, grid=grid(3840), stream=stream0)
        del primals_151
        buf195 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_6], Original ATen: [aten.mul]
        triton_poi_fused_mul_72.run(buf194, buf188, buf195, 188160, grid=grid(188160), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___14___block_3_0], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf197 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf196, buf197, 640, 49, grid=grid(640, 49), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf196  # reuse
        # Source Nodes: [result_21, result_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_73.run(buf197, primals_298, primals_299, primals_153, primals_154, buf180, buf198, 31360, grid=grid(31360), stream=stream0)
        del primals_154
        # Source Nodes: [getattr_l__mod___features___15___block_0_0], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf200 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___block_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf199, buf200, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf201 = reinterpret_tensor(buf199, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf199  # reuse
        buf202 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___block_0_1, getattr_l__mod___features___15___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68.run(buf200, primals_301, primals_302, primals_156, primals_157, buf201, buf202, 188160, grid=grid(188160), stream=stream0)
        del primals_157
        # Source Nodes: [getattr_l__mod___features___15___block_1_0], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_158, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf203, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf204 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___block_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf203, buf204, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf205 = reinterpret_tensor(buf203, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf203  # reuse
        buf206 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___block_1_1, getattr_l__mod___features___15___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_68.run(buf204, primals_304, primals_305, primals_159, primals_160, buf205, buf206, 188160, grid=grid(188160), stream=stream0)
        del primals_160
        buf207 = reinterpret_tensor(buf193, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf193  # reuse
        buf208 = reinterpret_tensor(buf207, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf207  # reuse
        # Source Nodes: [scale_35], Original ATen: [aten.mean]
        triton_per_fused_mean_69.run(buf208, buf206, 3840, 49, grid=grid(3840), stream=stream0)
        # Source Nodes: [scale_36], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 240, 1, 1), (240, 1, 1, 1))
        buf210 = reinterpret_tensor(buf209, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf209  # reuse
        # Source Nodes: [scale_36, scale_37], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_70.run(buf210, primals_162, 960, grid=grid(960), stream=stream0)
        del primals_162
        # Source Nodes: [scale_38], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 960, 1, 1), (960, 1, 1, 1))
        buf212 = empty_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf225 = empty_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_38, scale_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_71.run(buf211, primals_164, buf212, buf225, 3840, grid=grid(3840), stream=stream0)
        del primals_164
        buf213 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_7], Original ATen: [aten.mul]
        triton_poi_fused_mul_72.run(buf212, buf206, buf213, 188160, grid=grid(188160), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___15___block_3_0], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf215 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___block_3_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf214, buf215, 640, 49, grid=grid(640, 49), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf214  # reuse
        # Source Nodes: [result_23, result_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_73.run(buf215, primals_307, primals_308, primals_166, primals_167, buf198, buf216, 31360, grid=grid(31360), stream=stream0)
        del primals_167
        # Source Nodes: [l__mod___features_16_0], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf218 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_16_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf217, buf218, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf219 = reinterpret_tensor(buf217, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf217  # reuse
        # Source Nodes: [l__mod___features_16_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_74.run(buf218, primals_310, primals_311, primals_169, primals_170, buf219, 188160, grid=grid(188160), stream=stream0)
        del primals_170
        buf220 = reinterpret_tensor(buf211, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf211  # reuse
        buf221 = reinterpret_tensor(buf220, (4, 960), (960, 1), 0); del buf220  # reuse
        # Source Nodes: [x, x_1, x_2], Original ATen: [aten.hardswish, aten.mean, aten.view]
        triton_per_fused_hardswish_mean_view_75.run(buf221, buf219, 3840, 49, grid=grid(3840), stream=stream0)
        buf222 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf221, reinterpret_tensor(primals_171, (960, 1280), (1, 960), 0), alpha=1, beta=1, out=buf222)
        del primals_172
        buf223 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_76.run(buf222, buf223, 5120, grid=grid(5120), stream=stream0)
        buf224 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_174, buf223, reinterpret_tensor(primals_173, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf224)
        del primals_174
        buf228 = empty_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_77.run(buf157, primals_125, buf228, 2688, grid=grid(2688), stream=stream0)
        del buf157
        del primals_125
        buf229 = empty_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_18], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_78.run(buf138, primals_112, buf229, 1920, grid=grid(1920), stream=stream0)
        del buf138
        del primals_112
        buf230 = empty_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_13], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_79.run(buf75, primals_63, buf230, 480, grid=grid(480), stream=stream0)
        del buf75
        del primals_63
        buf231 = empty_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_8], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_79.run(buf58, primals_50, buf231, 480, grid=grid(480), stream=stream0)
        del buf58
        del primals_50
        buf232 = empty_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.bool)
        # Source Nodes: [scale_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_80.run(buf41, primals_37, buf232, 288, grid=grid(288), stream=stream0)
        del buf41
        del primals_37
        return (buf224, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, buf1, buf3, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf23, buf25, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf38, buf40, buf42, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf55, buf57, buf59, buf60, buf62, buf63, buf65, buf66, buf68, buf69, buf72, buf74, buf76, buf77, buf79, buf80, buf82, buf83, buf84, buf86, buf87, buf88, buf90, buf91, buf93, buf94, buf95, buf97, buf98, buf99, buf101, buf102, buf104, buf105, buf106, buf108, buf109, buf110, buf112, buf113, buf115, buf116, buf117, buf119, buf120, buf121, buf123, buf124, buf126, buf127, buf128, buf130, buf131, buf132, buf135, buf137, buf139, buf140, buf142, buf143, buf145, buf146, buf147, buf149, buf150, buf151, buf154, buf156, buf158, buf159, buf161, buf162, buf164, buf165, buf166, buf168, buf169, buf170, buf172, buf174, buf176, buf177, buf179, buf180, buf182, buf183, buf184, buf186, buf187, buf188, buf190, buf192, buf194, buf195, buf197, buf198, buf200, buf201, buf202, buf204, buf205, buf206, buf208, buf210, buf212, buf213, buf215, buf216, buf218, buf219, buf221, buf222, buf223, reinterpret_tensor(primals_173, (1000, 1280), (1280, 1), 0), reinterpret_tensor(primals_171, (1280, 960), (960, 1), 0), buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1280, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_199 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v3_large', benchmark_compiled_module)
