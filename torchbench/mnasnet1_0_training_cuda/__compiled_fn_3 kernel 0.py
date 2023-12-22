
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7df2wzqzj65kpbrw3kvwd46dfszadkdpcxhoozudelgek5bfog5.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddiafui2aenk2k6pegmmv3edykl5nk22qpi5k6jitm6skipivnd.py
# Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
# l__mod___layers_0 => convolution
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
    ynumel = 128
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sxtpzyryok754jst5vp3cakq2p4cad5ubiq6z3tn7duwin2ftw.py
# Source Nodes: [l__mod___layers_1, l__mod___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___layers_1 => add_1, mul_1, mul_2, sub
# l__mod___layers_2 => relu
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
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
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


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3qv64nlhnkrowse4r5cp56q3wbstz4ivbx6xl2epdtedgr4nyg.py
# Source Nodes: [l__mod___layers_6], Original ATen: [aten.convolution]
# l__mod___layers_6 => convolution_2
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/oh/coheutiprgkw2uglgnm7aquclxbx4c6cidepi4osqbnjsnoojbtu.py
# Source Nodes: [l__mod___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# l__mod___layers_7 => add_5, mul_7, mul_8, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': []},
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4ezc632nwg274b74manjjahbgsuqbiki3zo6tjzvmnjhs637x5.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___8_____0___layers_0 => convolution_3
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
    ynumel = 192
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (602112*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyzfxxu5krfuoz2l2y6ljhcgh3qmsieevaka7nmbsmxnbvwwpap.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_1, getattr_getattr_l__mod___layers___8_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____0___layers_1 => add_7, mul_10, mul_11, sub_3
# getattr_getattr_l__mod___layers___8_____0___layers_2 => relu_2
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
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cftc3k6vzknwieafthdbgnn265y7hsitdq34ydhfuwgxx6cbfp4v.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___8_____0___layers_3 => convolution_4
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
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkhi2w6z5yockfpsawo4bman74kosfke232tt2b5nhbm3k7djje.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_4, getattr_getattr_l__mod___layers___8_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____0___layers_4 => add_9, mul_13, mul_14, sub_4
# getattr_getattr_l__mod___layers___8_____0___layers_5 => relu_3
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
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3uhz76qk6cqihblerx6fxs5dpuzfsjees52yr7u2auccjprugb.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___8_____0___layers_6 => convolution_5
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


# kernel path: /tmp/torchinductor_youkaichao/yt/cytpzfudd2k3n7e3ts6bbaba67asnqlykzbflapt674nw7g2xp3g.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___8_____0___layers_7 => add_11, mul_16, mul_17, sub_5
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
    tmp4 = 1e-05
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
# Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___8_____1___layers_0 => convolution_6
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


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhiwxs773q3w7xyyoy7n3c4ew4b7zzmarjlnbncv6227dghnisz.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_1, getattr_getattr_l__mod___layers___8_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____1___layers_1 => add_13, mul_19, mul_20, sub_6
# getattr_getattr_l__mod___layers___8_____1___layers_2 => relu_4
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


# kernel path: /tmp/torchinductor_youkaichao/rt/crtqwc5cq6pkc6wtznhsazan7xoxs4upsxrtpiukbryt3tir6x77.py
# Source Nodes: [add, getattr_getattr_l__mod___layers___8_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add => add_18
# getattr_getattr_l__mod___layers___8_____1___layers_7 => add_17, mul_25, mul_26, sub_8
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dw7telp4vujuxl75ztq37pplsqcxabhm765q2svjofmjnasppw.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___9_____0___layers_3 => convolution_13
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


# kernel path: /tmp/torchinductor_youkaichao/rl/crlmhkei2lmfim5jktplzmw7qhssqekb4rd47mrfrfadbabkcfzi.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_4, getattr_getattr_l__mod___layers___9_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___9_____0___layers_4 => add_29, mul_40, mul_41, sub_13
# getattr_getattr_l__mod___layers___9_____0___layers_5 => relu_9
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


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hg4vinerlydtja2xkedgc7vn3lfdudqpsraphkupqyqabbhz22.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___9_____0___layers_6 => convolution_14
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coeh4ty46l6nis5nw57jneiun2lmwufngenlmcgcfbr265pkulon.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___9_____0___layers_7 => add_31, mul_43, mul_44, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bl/cbl6yuoxhfusau6uwtu7c35h5otdfwgeexf5n52kibonc3n34nek.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___9_____1___layers_0 => convolution_15
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgt4caasxu6ld7jqhkbfcc7pxkk3wpybqzqodiz4rlef6wbl5c4q.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_1, getattr_getattr_l__mod___layers___9_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___9_____1___layers_1 => add_33, mul_46, mul_47, sub_15
# getattr_getattr_l__mod___layers___9_____1___layers_2 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yb/cybixw2pjtf2ncqmp356kkdeig2jtebcyjfkx4l7pzbpmgybjra6.py
# Source Nodes: [add_2, getattr_getattr_l__mod___layers___9_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_2 => add_38
# getattr_getattr_l__mod___layers___9_____1___layers_7 => add_37, mul_52, mul_53, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvoov25goxk7cc7clqn3bc67kz4e5dzhzp44ma472m4covqc6rpf.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___10_____0___layers_0 => convolution_21
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hw/chwcfmzstzuazcvmfddmku5ugk2jcu7zi4frxd2hqbegwxnj2bss.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_1, getattr_getattr_l__mod___layers___10_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____0___layers_1 => add_47, mul_64, mul_65, sub_21
# getattr_getattr_l__mod___layers___10_____0___layers_2 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkh5o4zj3hdehc63esxbt6prfmiecxvksh4uahqiunbhmo3fugs.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___10_____0___layers_3 => convolution_22
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5btnag6zci3h4fxja4kr2ldcyaeh42gj2kn55hnumjhvla4dbc.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_4, getattr_getattr_l__mod___layers___10_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____0___layers_4 => add_49, mul_67, mul_68, sub_22
# getattr_getattr_l__mod___layers___10_____0___layers_5 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/pp/cppuv4lvhke3n3sexevsyztl2uzfp72z3ajid2d3w42s7fy7zeqy.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___10_____0___layers_6 => convolution_23
triton_poi_fused_convolution_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ee/ceer2nsytaqhlx7brn56nr6vr7enh7txqwyd5tqnp4kjgryf5j3h.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___10_____0___layers_7 => add_51, mul_70, mul_71, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2a/c2aumqre3ohxuk4opgv4bmfb2lonkntrp7dqmlolf36znd3ztsnx.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___10_____1___layers_0 => convolution_24
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ic/cic6tatp6djah6xjtr2xzmndr6p2qkii5wldauzg4quystc54isz.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_1, getattr_getattr_l__mod___layers___10_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____1___layers_1 => add_53, mul_73, mul_74, sub_24
# getattr_getattr_l__mod___layers___10_____1___layers_2 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzdkemnp5gn2v6pbcmebfynbxdoiuw7d7r3cn5qj3eylgdidbt7.py
# Source Nodes: [add_4, getattr_getattr_l__mod___layers___10_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_4 => add_58
# getattr_getattr_l__mod___layers___10_____1___layers_7 => add_57, mul_79, mul_80, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_add_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_30', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfoedxfvphgtmxa37ejpfl7movtmxbxtgswqiorovmidvil65pc.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___11_____0___layers_6 => convolution_32
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (18816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmi7ea2vw7ck6iawhtes7fcqxjdcg53i4dkvjnlnoeeqi2baxme.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___11_____0___layers_7 => add_71, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_youkaichao/jj/cjj4v37kks4wdt46paqeukwhz2pr4er6ckihzszxgjctqddbcgcr.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___11_____1___layers_0 => convolution_33
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (576*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6swgxkchkmfmci2bydlbkthpkfmqh66bjbyakhge5pb44meg34.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_1, getattr_getattr_l__mod___layers___11_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___11_____1___layers_1 => add_73, mul_100, mul_101, sub_33
# getattr_getattr_l__mod___layers___11_____1___layers_2 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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


# kernel path: /tmp/torchinductor_youkaichao/vi/cvinjodz5l24dh6ehvaxkabr2qn4ohvd7kmv2w2bc5wuxiip2lty.py
# Source Nodes: [add_6, getattr_getattr_l__mod___layers___11_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_6 => add_78
# getattr_getattr_l__mod___layers___11_____1___layers_7 => add_77, mul_106, mul_107, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxfgkscqlushmgzejkmy6llhosgruud7xruuk3temevvwu2czmy.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___12_____0___layers_3 => convolution_37
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (576*x2) + (28224*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyfjlicz5pd3eu2xtegasa2qv6tn363roly6mpxgcqfveazdeao.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_4, getattr_getattr_l__mod___layers___12_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___12_____0___layers_4 => add_82, mul_112, mul_113, sub_37
# getattr_getattr_l__mod___layers___12_____0___layers_5 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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


# kernel path: /tmp/torchinductor_youkaichao/zq/czqhcawjjlhhcfav5v4pzfkubmw62mh53ax6mfdncpnc44xiisbb.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___12_____0___layers_6 => convolution_38
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cju3qydguyiux2ytoyu777osfxwroywq6aomnpcb32y5aom5uerk.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___12_____0___layers_7 => add_84, mul_115, mul_116, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjsobk4if6hgcbrdvv43qsvnjbfdafp7zj2h34fws6gkdp2627cj.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___12_____1___layers_0 => convolution_39
triton_poi_fused_convolution_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23bmm2ulyez6k72xfo2ifcwpqpdrhcggwg5xu6xpmsmtlwkabs5.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_1, getattr_getattr_l__mod___layers___12_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___12_____1___layers_1 => add_86, mul_118, mul_119, sub_39
# getattr_getattr_l__mod___layers___12_____1___layers_2 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6ti6osmg3ylghwt7x2p5nztsdoka6uf32zw6f4lbii6fsne63lp.py
# Source Nodes: [add_7, getattr_getattr_l__mod___layers___12_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_7 => add_91
# getattr_getattr_l__mod___layers___12_____1___layers_7 => add_90, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_add_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3hnn3k2fllq4cjpjsjb27kqy4rdznxnvns573pjcpx5jfjyvqz3.py
# Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_6], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___layers___13_____0___layers_6 => convolution_50
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctwezweksrl3cyiqvjq7h2msffe4d2ikc2ve4oo77ayqfgclqns.py
# Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___13_____0___layers_7 => add_111, mul_151, mul_152, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
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


# kernel path: /tmp/torchinductor_youkaichao/7t/c7txvbadp3vozt2cew4ohvbuiq47ys4lccc4ymgycfqn5bjgae3y.py
# Source Nodes: [l__mod___layers_14], Original ATen: [aten.convolution]
# l__mod___layers_14 => convolution_51
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (62720*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cva6czylhcov7l2j75kzh7zi7qbromzklzhysdl4jxczagjkpn4m.py
# Source Nodes: [l__mod___layers_15, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# l__mod___layers_15 => add_113, mul_154, mul_155, sub_51
# x => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
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


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgu76ic5rn5oimshboramqabhnzbg6wnozhhuvvbb65qzwhxokn.py
# Source Nodes: [x_1], Original ATen: [aten.mean]
# x_1 => mean
triton_per_fused_mean_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_12, (48, ), (1, ))
    assert_size_stride(primals_13, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_15, (48, ), (1, ))
    assert_size_stride(primals_16, (24, 48, 1, 1), (48, 1, 1, 1))
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
    assert_size_stride(primals_31, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_33, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (24, ), (1, ))
    assert_size_stride(primals_37, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (72, ), (1, ))
    assert_size_stride(primals_39, (72, ), (1, ))
    assert_size_stride(primals_40, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_41, (72, ), (1, ))
    assert_size_stride(primals_42, (72, ), (1, ))
    assert_size_stride(primals_43, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_44, (40, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_46, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_48, (120, ), (1, ))
    assert_size_stride(primals_49, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_51, (120, ), (1, ))
    assert_size_stride(primals_52, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_54, (40, ), (1, ))
    assert_size_stride(primals_55, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_57, (120, ), (1, ))
    assert_size_stride(primals_58, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_60, (120, ), (1, ))
    assert_size_stride(primals_61, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (40, ), (1, ))
    assert_size_stride(primals_63, (40, ), (1, ))
    assert_size_stride(primals_64, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_65, (240, ), (1, ))
    assert_size_stride(primals_66, (240, ), (1, ))
    assert_size_stride(primals_67, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_69, (240, ), (1, ))
    assert_size_stride(primals_70, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_71, (80, ), (1, ))
    assert_size_stride(primals_72, (80, ), (1, ))
    assert_size_stride(primals_73, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_74, (480, ), (1, ))
    assert_size_stride(primals_75, (480, ), (1, ))
    assert_size_stride(primals_76, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (480, ), (1, ))
    assert_size_stride(primals_78, (480, ), (1, ))
    assert_size_stride(primals_79, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_80, (80, ), (1, ))
    assert_size_stride(primals_81, (80, ), (1, ))
    assert_size_stride(primals_82, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_83, (480, ), (1, ))
    assert_size_stride(primals_84, (480, ), (1, ))
    assert_size_stride(primals_85, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_87, (480, ), (1, ))
    assert_size_stride(primals_88, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_89, (80, ), (1, ))
    assert_size_stride(primals_90, (80, ), (1, ))
    assert_size_stride(primals_91, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_93, (480, ), (1, ))
    assert_size_stride(primals_94, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (480, ), (1, ))
    assert_size_stride(primals_96, (480, ), (1, ))
    assert_size_stride(primals_97, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_98, (96, ), (1, ))
    assert_size_stride(primals_99, (96, ), (1, ))
    assert_size_stride(primals_100, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_101, (576, ), (1, ))
    assert_size_stride(primals_102, (576, ), (1, ))
    assert_size_stride(primals_103, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (576, ), (1, ))
    assert_size_stride(primals_105, (576, ), (1, ))
    assert_size_stride(primals_106, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_107, (96, ), (1, ))
    assert_size_stride(primals_108, (96, ), (1, ))
    assert_size_stride(primals_109, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (576, ), (1, ))
    assert_size_stride(primals_111, (576, ), (1, ))
    assert_size_stride(primals_112, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_114, (576, ), (1, ))
    assert_size_stride(primals_115, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_119, (1152, ), (1, ))
    assert_size_stride(primals_120, (1152, ), (1, ))
    assert_size_stride(primals_121, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (1152, ), (1, ))
    assert_size_stride(primals_123, (1152, ), (1, ))
    assert_size_stride(primals_124, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_126, (192, ), (1, ))
    assert_size_stride(primals_127, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (1152, ), (1, ))
    assert_size_stride(primals_129, (1152, ), (1, ))
    assert_size_stride(primals_130, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_132, (1152, ), (1, ))
    assert_size_stride(primals_133, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_137, (1152, ), (1, ))
    assert_size_stride(primals_138, (1152, ), (1, ))
    assert_size_stride(primals_139, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (1152, ), (1, ))
    assert_size_stride(primals_141, (1152, ), (1, ))
    assert_size_stride(primals_142, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_146, (1152, ), (1, ))
    assert_size_stride(primals_147, (1152, ), (1, ))
    assert_size_stride(primals_148, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (1152, ), (1, ))
    assert_size_stride(primals_150, (1152, ), (1, ))
    assert_size_stride(primals_151, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_152, (320, ), (1, ))
    assert_size_stride(primals_153, (320, ), (1, ))
    assert_size_stride(primals_154, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_155, (1280, ), (1, ))
    assert_size_stride(primals_156, (1280, ), (1, ))
    assert_size_stride(primals_157, (1000, 1280), (1280, 1))
    assert_size_stride(primals_158, (1000, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (), ())
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (48, ), (1, ))
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (48, ), (1, ))
    assert_size_stride(primals_172, (48, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (72, ), (1, ))
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (72, ), (1, ))
    assert_size_stride(primals_181, (72, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (72, ), (1, ))
    assert_size_stride(primals_187, (72, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (72, ), (1, ))
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (72, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (40, ), (1, ))
    assert_size_stride(primals_202, (40, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (120, ), (1, ))
    assert_size_stride(primals_205, (120, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (120, ), (1, ))
    assert_size_stride(primals_208, (120, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (40, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (120, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (40, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (240, ), (1, ))
    assert_size_stride(primals_223, (240, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (240, ), (1, ))
    assert_size_stride(primals_226, (240, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (80, ), (1, ))
    assert_size_stride(primals_229, (80, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (480, ), (1, ))
    assert_size_stride(primals_241, (480, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (96, ), (1, ))
    assert_size_stride(primals_256, (96, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (576, ), (1, ))
    assert_size_stride(primals_259, (576, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (576, ), (1, ))
    assert_size_stride(primals_262, (576, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (96, ), (1, ))
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (576, ), (1, ))
    assert_size_stride(primals_268, (576, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (576, ), (1, ))
    assert_size_stride(primals_271, (576, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (1152, ), (1, ))
    assert_size_stride(primals_277, (1152, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1152, ), (1, ))
    assert_size_stride(primals_280, (1152, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (192, ), (1, ))
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (1152, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (192, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (1152, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (1152, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (1280, ), (1, ))
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_315, buf1, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_315
        # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf4 = reinterpret_tensor(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        # Source Nodes: [l__mod___layers_1, l__mod___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf3, primals_159, primals_160, primals_2, primals_3, buf4, 1605632, grid=grid(1605632), stream=stream0)
        del primals_3
        # Source Nodes: [l__mod___layers_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf6 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf5, buf6, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf7 = reinterpret_tensor(buf5, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf5  # reuse
        # Source Nodes: [l__mod___layers_4, l__mod___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf6, primals_162, primals_163, primals_5, primals_6, buf7, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [l__mod___layers_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf9 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf8, buf9, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf10 = reinterpret_tensor(buf8, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf8  # reuse
        # Source Nodes: [l__mod___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf9, primals_165, primals_166, primals_8, primals_9, buf10, 802816, grid=grid(802816), stream=stream0)
        del primals_9
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_0], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 48, 112, 112), (602112, 12544, 112, 1))
        buf12 = empty_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf11, buf12, 192, 12544, grid=grid(192, 12544), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (4, 48, 112, 112), (602112, 1, 5376, 48), 0); del buf11  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_1, getattr_getattr_l__mod___layers___8_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, primals_168, primals_169, primals_11, primals_12, buf13, 2408448, grid=grid(2408448), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_3], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf14, (4, 48, 56, 56), (150528, 3136, 56, 1))
        buf15 = empty_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf14, buf15, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (4, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf14  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_4, getattr_getattr_l__mod___layers___8_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf15, primals_171, primals_172, primals_14, primals_15, buf16, 602112, grid=grid(602112), stream=stream0)
        del primals_15
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_6], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf18 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf17, buf18, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf17  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf18, primals_174, primals_175, primals_17, primals_18, buf19, 301056, grid=grid(301056), stream=stream0)
        del primals_18
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf21 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf20, buf21, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf22 = reinterpret_tensor(buf20, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_1, getattr_getattr_l__mod___layers___8_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf21, primals_177, primals_178, primals_20, primals_21, buf22, 903168, grid=grid(903168), stream=stream0)
        del primals_21
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_3], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf23, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf24 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf23, buf24, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf25 = reinterpret_tensor(buf23, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf23  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_4, getattr_getattr_l__mod___layers___8_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf24, primals_180, primals_181, primals_23, primals_24, buf25, 903168, grid=grid(903168), stream=stream0)
        del primals_24
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_6], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf27 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf26, buf27, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf28 = reinterpret_tensor(buf26, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf26  # reuse
        # Source Nodes: [add, getattr_getattr_l__mod___layers___8_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf27, primals_183, primals_184, primals_26, primals_27, buf19, buf28, 301056, grid=grid(301056), stream=stream0)
        del primals_27
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_0], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf30 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf29, buf30, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf29  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_1, getattr_getattr_l__mod___layers___8_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf30, primals_186, primals_187, primals_29, primals_30, buf31, 903168, grid=grid(903168), stream=stream0)
        del primals_30
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_3], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf32, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf33 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf32, buf33, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf34 = reinterpret_tensor(buf32, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf32  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_4, getattr_getattr_l__mod___layers___8_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf33, primals_189, primals_190, primals_32, primals_33, buf34, 903168, grid=grid(903168), stream=stream0)
        del primals_33
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_6], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf36 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf35, buf36, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf35  # reuse
        # Source Nodes: [add_1, getattr_getattr_l__mod___layers___8_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf36, primals_192, primals_193, primals_35, primals_36, buf28, buf37, 301056, grid=grid(301056), stream=stream0)
        del primals_36
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_0], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 72, 56, 56), (225792, 3136, 56, 1))
        buf39 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf38, buf39, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (4, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf38  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_1, getattr_getattr_l__mod___layers___9_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf39, primals_195, primals_196, primals_38, primals_39, buf40, 903168, grid=grid(903168), stream=stream0)
        del primals_39
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_40, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf41, (4, 72, 28, 28), (56448, 784, 28, 1))
        buf42 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf41, buf42, 288, 784, grid=grid(288, 784), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (4, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_4, getattr_getattr_l__mod___layers___9_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf42, primals_198, primals_199, primals_41, primals_42, buf43, 225792, grid=grid(225792), stream=stream0)
        del primals_42
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_6], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf45 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf44, buf45, 160, 784, grid=grid(160, 784), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf44  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf45, primals_201, primals_202, primals_44, primals_45, buf46, 125440, grid=grid(125440), stream=stream0)
        del primals_45
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_0], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf48 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf47, buf48, 480, 784, grid=grid(480, 784), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf47  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_1, getattr_getattr_l__mod___layers___9_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf48, primals_204, primals_205, primals_47, primals_48, buf49, 376320, grid=grid(376320), stream=stream0)
        del primals_48
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_3], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_49, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf50, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf51 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf50, buf51, 480, 784, grid=grid(480, 784), stream=stream0)
        buf52 = reinterpret_tensor(buf50, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf50  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_4, getattr_getattr_l__mod___layers___9_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf51, primals_207, primals_208, primals_50, primals_51, buf52, 376320, grid=grid(376320), stream=stream0)
        del primals_51
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_6], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf54 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf53, buf54, 160, 784, grid=grid(160, 784), stream=stream0)
        buf55 = reinterpret_tensor(buf53, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf53  # reuse
        # Source Nodes: [add_2, getattr_getattr_l__mod___layers___9_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf54, primals_210, primals_211, primals_53, primals_54, buf46, buf55, 125440, grid=grid(125440), stream=stream0)
        del primals_54
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf57 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf56, buf57, 480, 784, grid=grid(480, 784), stream=stream0)
        buf58 = reinterpret_tensor(buf56, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf56  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_1, getattr_getattr_l__mod___layers___9_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf57, primals_213, primals_214, primals_56, primals_57, buf58, 376320, grid=grid(376320), stream=stream0)
        del primals_57
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_3], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_58, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf59, (4, 120, 28, 28), (94080, 784, 28, 1))
        buf60 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf59, buf60, 480, 784, grid=grid(480, 784), stream=stream0)
        buf61 = reinterpret_tensor(buf59, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf59  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_4, getattr_getattr_l__mod___layers___9_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf60, primals_216, primals_217, primals_59, primals_60, buf61, 376320, grid=grid(376320), stream=stream0)
        del primals_60
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_6], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf63 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf62, buf63, 160, 784, grid=grid(160, 784), stream=stream0)
        buf64 = reinterpret_tensor(buf62, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf62  # reuse
        # Source Nodes: [add_3, getattr_getattr_l__mod___layers___9_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf63, primals_219, primals_220, primals_62, primals_63, buf55, buf64, 125440, grid=grid(125440), stream=stream0)
        del primals_63
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_0], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 240, 28, 28), (188160, 784, 28, 1))
        buf66 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf65, buf66, 960, 784, grid=grid(960, 784), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (4, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf65  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_1, getattr_getattr_l__mod___layers___10_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf66, primals_222, primals_223, primals_65, primals_66, buf67, 752640, grid=grid(752640), stream=stream0)
        del primals_66
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_3], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_67, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf68, (4, 240, 14, 14), (47040, 196, 14, 1))
        buf69 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf68, buf69, 960, 196, grid=grid(960, 196), stream=stream0)
        buf70 = reinterpret_tensor(buf68, (4, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf68  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_4, getattr_getattr_l__mod___layers___10_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf69, primals_225, primals_226, primals_68, primals_69, buf70, 188160, grid=grid(188160), stream=stream0)
        del primals_69
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_6], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf72 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf71, buf72, 320, 196, grid=grid(320, 196), stream=stream0)
        buf73 = reinterpret_tensor(buf71, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf72, primals_228, primals_229, primals_71, primals_72, buf73, 62720, grid=grid(62720), stream=stream0)
        del primals_72
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_0], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf75 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf74, buf75, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf76 = reinterpret_tensor(buf74, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf74  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_1, getattr_getattr_l__mod___layers___10_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf75, primals_231, primals_232, primals_74, primals_75, buf76, 376320, grid=grid(376320), stream=stream0)
        del primals_75
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_3], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_76, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf77, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf78 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf77, buf78, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf79 = reinterpret_tensor(buf77, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf77  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_4, getattr_getattr_l__mod___layers___10_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf78, primals_234, primals_235, primals_77, primals_78, buf79, 376320, grid=grid(376320), stream=stream0)
        del primals_78
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_6], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf81 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf80, buf81, 320, 196, grid=grid(320, 196), stream=stream0)
        buf82 = reinterpret_tensor(buf80, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf80  # reuse
        # Source Nodes: [add_4, getattr_getattr_l__mod___layers___10_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_30.run(buf81, primals_237, primals_238, primals_80, primals_81, buf73, buf82, 62720, grid=grid(62720), stream=stream0)
        del primals_81
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_0], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf84 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf83, buf84, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf85 = reinterpret_tensor(buf83, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf83  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_1, getattr_getattr_l__mod___layers___10_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf84, primals_240, primals_241, primals_83, primals_84, buf85, 376320, grid=grid(376320), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_3], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_85, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf86, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf87 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf86, buf87, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf86  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_4, getattr_getattr_l__mod___layers___10_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf87, primals_243, primals_244, primals_86, primals_87, buf88, 376320, grid=grid(376320), stream=stream0)
        del primals_87
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_6], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf90 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf89, buf90, 320, 196, grid=grid(320, 196), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf89  # reuse
        # Source Nodes: [add_5, getattr_getattr_l__mod___layers___10_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_30.run(buf90, primals_246, primals_247, primals_89, primals_90, buf82, buf91, 62720, grid=grid(62720), stream=stream0)
        del primals_90
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_0], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf93 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf92, buf93, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf92  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_1, getattr_getattr_l__mod___layers___11_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf93, primals_249, primals_250, primals_92, primals_93, buf94, 376320, grid=grid(376320), stream=stream0)
        del primals_93
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_3], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf95, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf96 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf95, buf96, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf97 = reinterpret_tensor(buf95, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_4, getattr_getattr_l__mod___layers___11_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf96, primals_252, primals_253, primals_95, primals_96, buf97, 376320, grid=grid(376320), stream=stream0)
        del primals_96
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_6], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 96, 14, 14), (18816, 196, 14, 1))
        buf99 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf98, buf99, 384, 196, grid=grid(384, 196), stream=stream0)
        buf100 = reinterpret_tensor(buf98, (4, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf98  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf99, primals_255, primals_256, primals_98, primals_99, buf100, 75264, grid=grid(75264), stream=stream0)
        del primals_99
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_0], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf102 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf101, buf102, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf101  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_1, getattr_getattr_l__mod___layers___11_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf102, primals_258, primals_259, primals_101, primals_102, buf103, 451584, grid=grid(451584), stream=stream0)
        del primals_102
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_3], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf104, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf105 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf104, buf105, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf106 = reinterpret_tensor(buf104, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf104  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_4, getattr_getattr_l__mod___layers___11_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf105, primals_261, primals_262, primals_104, primals_105, buf106, 451584, grid=grid(451584), stream=stream0)
        del primals_105
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_6], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 96, 14, 14), (18816, 196, 14, 1))
        buf108 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf107, buf108, 384, 196, grid=grid(384, 196), stream=stream0)
        buf109 = reinterpret_tensor(buf107, (4, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf107  # reuse
        # Source Nodes: [add_6, getattr_getattr_l__mod___layers___11_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_35.run(buf108, primals_264, primals_265, primals_107, primals_108, buf100, buf109, 75264, grid=grid(75264), stream=stream0)
        del primals_108
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_0], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf111 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf110, buf111, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf110  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_1, getattr_getattr_l__mod___layers___12_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf111, primals_267, primals_268, primals_110, primals_111, buf112, 451584, grid=grid(451584), stream=stream0)
        del primals_111
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_3], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_112, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf113, (4, 576, 7, 7), (28224, 49, 7, 1))
        buf114 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf113, buf114, 2304, 49, grid=grid(2304, 49), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (4, 576, 7, 7), (28224, 1, 4032, 576), 0); del buf113  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_4, getattr_getattr_l__mod___layers___12_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf114, primals_270, primals_271, primals_113, primals_114, buf115, 112896, grid=grid(112896), stream=stream0)
        del primals_114
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_6], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf117 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf116, buf117, 768, 49, grid=grid(768, 49), stream=stream0)
        buf118 = reinterpret_tensor(buf116, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf116  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf117, primals_273, primals_274, primals_116, primals_117, buf118, 37632, grid=grid(37632), stream=stream0)
        del primals_117
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_0], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf120 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf119, buf120, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf121 = reinterpret_tensor(buf119, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf119  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_1, getattr_getattr_l__mod___layers___12_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf120, primals_276, primals_277, primals_119, primals_120, buf121, 225792, grid=grid(225792), stream=stream0)
        del primals_120
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_3], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_121, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf122, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf123 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf122, buf123, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf122  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_4, getattr_getattr_l__mod___layers___12_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf123, primals_279, primals_280, primals_122, primals_123, buf124, 225792, grid=grid(225792), stream=stream0)
        del primals_123
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_6], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf126 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf125, buf126, 768, 49, grid=grid(768, 49), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf125  # reuse
        # Source Nodes: [add_7, getattr_getattr_l__mod___layers___12_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf126, primals_282, primals_283, primals_125, primals_126, buf118, buf127, 37632, grid=grid(37632), stream=stream0)
        del primals_126
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_0], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf129 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf128, buf129, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf128  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_1, getattr_getattr_l__mod___layers___12_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf129, primals_285, primals_286, primals_128, primals_129, buf130, 225792, grid=grid(225792), stream=stream0)
        del primals_129
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_3], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_130, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf131, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf132 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf131, buf132, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf131  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_4, getattr_getattr_l__mod___layers___12_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf132, primals_288, primals_289, primals_131, primals_132, buf133, 225792, grid=grid(225792), stream=stream0)
        del primals_132
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_6], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf135 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf134, buf135, 768, 49, grid=grid(768, 49), stream=stream0)
        buf136 = reinterpret_tensor(buf134, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf134  # reuse
        # Source Nodes: [add_8, getattr_getattr_l__mod___layers___12_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf135, primals_291, primals_292, primals_134, primals_135, buf127, buf136, 37632, grid=grid(37632), stream=stream0)
        del primals_135
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_0], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf138 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf137, buf138, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf139 = reinterpret_tensor(buf137, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf137  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_1, getattr_getattr_l__mod___layers___12_____3___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf138, primals_294, primals_295, primals_137, primals_138, buf139, 225792, grid=grid(225792), stream=stream0)
        del primals_138
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_3], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_139, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf140, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf141 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf140, buf141, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf142 = reinterpret_tensor(buf140, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf140  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_4, getattr_getattr_l__mod___layers___12_____3___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf141, primals_297, primals_298, primals_140, primals_141, buf142, 225792, grid=grid(225792), stream=stream0)
        del primals_141
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_6], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf144 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf143, buf144, 768, 49, grid=grid(768, 49), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf143  # reuse
        # Source Nodes: [add_9, getattr_getattr_l__mod___layers___12_____3___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf144, primals_300, primals_301, primals_143, primals_144, buf136, buf145, 37632, grid=grid(37632), stream=stream0)
        del primals_144
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_0], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf147 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf146, buf147, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf148 = reinterpret_tensor(buf146, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf146  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_1, getattr_getattr_l__mod___layers___13_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf147, primals_303, primals_304, primals_146, primals_147, buf148, 225792, grid=grid(225792), stream=stream0)
        del primals_147
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_3], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf149, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf150 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf149, buf150, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf149  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_4, getattr_getattr_l__mod___layers___13_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf150, primals_306, primals_307, primals_149, primals_150, buf151, 225792, grid=grid(225792), stream=stream0)
        del primals_150
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_6], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 320, 7, 7), (15680, 49, 7, 1))
        buf153 = empty_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf152, buf153, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf154 = reinterpret_tensor(buf152, (4, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf152  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf153, primals_309, primals_310, primals_152, primals_153, buf154, 62720, grid=grid(62720), stream=stream0)
        del primals_153
        # Source Nodes: [l__mod___layers_14], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 1280, 7, 7), (62720, 49, 7, 1))
        buf156 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_14], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf155, buf156, 5120, 49, grid=grid(5120, 49), stream=stream0)
        buf157 = reinterpret_tensor(buf155, (4, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf155  # reuse
        buf161 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___layers_15, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_46.run(buf156, primals_312, primals_313, primals_155, primals_156, buf157, buf161, 250880, grid=grid(250880), stream=stream0)
        del primals_156
        buf158 = empty((4, 1280), device='cuda', dtype=torch.float32)
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.mean]
        triton_per_fused_mean_47.run(buf159, buf157, 5120, 49, grid=grid(5120), stream=stream0)
        del buf157
        buf160 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf159, reinterpret_tensor(primals_157, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf160)
        del primals_158
        return (buf160, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf22, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf67, buf69, buf70, buf72, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf84, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf121, buf123, buf124, buf126, buf127, buf129, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf142, buf144, buf145, buf147, buf148, buf150, buf151, buf153, buf154, buf156, buf159, reinterpret_tensor(primals_157, (1000, 1280), (1280, 1), 0), buf161, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_31 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_162 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet1_0', benchmark_compiled_module)
