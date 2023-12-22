
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


# kernel path: /tmp/torchinductor_youkaichao/5f/c5fut5xp53hbaiooisteulvhzadimlaz6ycbw3il5nrftls3cpry.py
# Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# l__mod___features_0_1 => add_1, mul_1, mul_2, sub
# l__mod___features_0_2 => clamp_max, clamp_min
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3qv64nlhnkrowse4r5cp56q3wbstz4ivbx6xl2epdtedgr4nyg.py
# Source Nodes: [getattr_l__mod___features___1___conv_1], Original ATen: [aten.convolution]
# getattr_l__mod___features___1___conv_1 => convolution_2
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
# Source Nodes: [getattr_l__mod___features___1___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___1___conv_2 => add_5, mul_7, mul_8, sub_2
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


# kernel path: /tmp/torchinductor_youkaichao/wi/cwipxnwkxzs5oxv7f2lj3oyejrvml33xtejsinky4riqevteiwtr.py
# Source Nodes: [getattr_l__mod___features___2___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___conv_0_0 => convolution_3
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfrore44ybo2q7bibfcxmjxhr5ni7kvcooybaxqgiurfm3hkmdg.py
# Source Nodes: [getattr_l__mod___features___2___conv_0_1, getattr_l__mod___features___2___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___2___conv_0_1 => add_7, mul_10, mul_11, sub_3
# getattr_l__mod___features___2___conv_0_2 => clamp_max_2, clamp_min_2
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/colzwumsxzpxj5p2zb3oponcvq64f4rmmpzvjs6zpc4zmjqecwqg.py
# Source Nodes: [getattr_l__mod___features___2___conv_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___conv_1_0 => convolution_4
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5spz35spkz6usq5obu7blze3di2f56mjon6nrzl7du7szikaxou.py
# Source Nodes: [getattr_l__mod___features___2___conv_1_1, getattr_l__mod___features___2___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___2___conv_1_1 => add_9, mul_13, mul_14, sub_4
# getattr_l__mod___features___2___conv_1_2 => clamp_max_3, clamp_min_3
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3uhz76qk6cqihblerx6fxs5dpuzfsjees52yr7u2auccjprugb.py
# Source Nodes: [getattr_l__mod___features___2___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___2___conv_2 => convolution_5
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
# Source Nodes: [getattr_l__mod___features___2___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___2___conv_3 => add_11, mul_16, mul_17, sub_5
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czol4otxqryll45xsziapd3gphgoicthflyc4misulm33j66f7hx.py
# Source Nodes: [getattr_l__mod___features___3___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___3___conv_0_0 => convolution_6
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
    ynumel = 576
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cyk4rtsrb7zhjzzwnmvkfiins45gketntb36jgjlbms5avqehsd5.py
# Source Nodes: [getattr_l__mod___features___3___conv_0_1, getattr_l__mod___features___3___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___3___conv_0_1 => add_13, mul_19, mul_20, sub_6
# getattr_l__mod___features___3___conv_0_2 => clamp_max_4, clamp_min_4
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfc3apmjcjsardphltx7msin4pdrs2x4l7sk4hmsthntgvzkye2.py
# Source Nodes: [add, getattr_l__mod___features___3___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add => add_18
# getattr_l__mod___features___3___conv_3 => add_17, mul_25, mul_26, sub_8
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
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iguifybaieqvjsks3urdixrqdlksh7jd234tozaicqr7pwxop7.py
# Source Nodes: [getattr_l__mod___features___4___conv_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___4___conv_1_0 => convolution_10
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkth4agqvmnh6vw6ylctb24xmxfbrddajbpjwlogdgxd2jynsml.py
# Source Nodes: [getattr_l__mod___features___4___conv_1_1, getattr_l__mod___features___4___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___4___conv_1_1 => add_22, mul_31, mul_32, sub_10
# getattr_l__mod___features___4___conv_1_2 => clamp_max_7, clamp_min_7
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coip5qcalyfj3c3qrrcejyfhpkhsl6iwieopre4ogdi75zgsnq34.py
# Source Nodes: [getattr_l__mod___features___4___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___4___conv_2 => convolution_11
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3qtp6udxwt5qevb7sg3mgkra2slgnhbd23u6hi2v7zmjewt3ab.py
# Source Nodes: [getattr_l__mod___features___4___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___4___conv_3 => add_24, mul_34, mul_35, sub_11
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
    xnumel = 100352
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvwaxpmd3ura5t6c5jrjudzprngbynuqrfb37cwclnhc3dgbqxf.py
# Source Nodes: [getattr_l__mod___features___5___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___5___conv_0_0 => convolution_12
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lnp64jdyclhssb25um6whgatx3n725ce5ic4jmot6qq4u3tya7.py
# Source Nodes: [getattr_l__mod___features___5___conv_0_1, getattr_l__mod___features___5___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___5___conv_0_1 => add_26, mul_37, mul_38, sub_12
# getattr_l__mod___features___5___conv_0_2 => clamp_max_8, clamp_min_8
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp23voxtyw33mzadlmk63ro24u5uz4vvhh6tgzqnk6emiccmvh6i.py
# Source Nodes: [add_1, getattr_l__mod___features___5___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_1 => add_31
# getattr_l__mod___features___5___conv_3 => add_30, mul_43, mul_44, sub_14
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
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjel3uov3mxbmvixbyjyb3xfftpkuimlbljbzn4hzc2647lsjqrp.py
# Source Nodes: [getattr_l__mod___features___7___conv_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___7___conv_1_0 => convolution_19
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (37632*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4nz4edrq27ynu2rdyy4wkufcn62nbpi4hqcljft7piexnkc6p2.py
# Source Nodes: [getattr_l__mod___features___7___conv_1_1, getattr_l__mod___features___7___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___7___conv_1_1 => add_42, mul_58, mul_59, sub_19
# getattr_l__mod___features___7___conv_1_2 => clamp_max_13, clamp_min_13
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cdu2m3t2pqeixhykfugy5f5y27mrctko2kfrydeoejinp7dgpdw5.py
# Source Nodes: [getattr_l__mod___features___7___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___7___conv_2 => convolution_20
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (12544*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2ew35wkwpzgbdnzlule7rjhrhriu3tmlxw5x7urunjliljuimr.py
# Source Nodes: [getattr_l__mod___features___7___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___7___conv_3 => add_44, mul_61, mul_62, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5q6b7afd7mxthtyovpxxivcalietiqcvrte56xktwqv5s7edzg.py
# Source Nodes: [getattr_l__mod___features___8___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___8___conv_0_0 => convolution_21
triton_poi_fused_convolution_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6vytsk4cs7ncwhfr66t2olat3uuzf6rzecxu3qb7zc3lcctecf.py
# Source Nodes: [getattr_l__mod___features___8___conv_0_1, getattr_l__mod___features___8___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___8___conv_0_1 => add_46, mul_64, mul_65, sub_21
# getattr_l__mod___features___8___conv_0_2 => clamp_max_14, clamp_min_14
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg44bp2c43oso2xazxxm7se4awrwsdrt2upthuoipilct3ws7f2g.py
# Source Nodes: [add_3, getattr_l__mod___features___8___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_3 => add_51
# getattr_l__mod___features___8___conv_3 => add_50, mul_70, mul_71, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_add_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctr5ygwbizeldgivh52lmyhiwaxbh6cuqrhv4wtxw4jnkssahagj.py
# Source Nodes: [getattr_l__mod___features___11___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___11___conv_2 => convolution_32
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaa7gc6wklb43f7563uwvdypmqa2by6e45ebtwa2cpurhhvlvao.py
# Source Nodes: [getattr_l__mod___features___11___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___11___conv_3 => add_71, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3daykj6lqjkvf6fh66twvjb3dikgzpi7ivlvkox2ssa544of5a.py
# Source Nodes: [getattr_l__mod___features___12___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___12___conv_0_0 => convolution_33
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmuwixkhjkblxmmn7dvccittpfszbdpllsto4t4e4dr3ixpninb.py
# Source Nodes: [getattr_l__mod___features___12___conv_0_1, getattr_l__mod___features___12___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___12___conv_0_1 => add_73, mul_100, mul_101, sub_33
# getattr_l__mod___features___12___conv_0_2 => clamp_max_22, clamp_min_22
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74x4bmlksyx7wly2dp5di2h3vnudxk36ivvjex6hc5am623mif3.py
# Source Nodes: [add_6, getattr_l__mod___features___12___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_6 => add_78
# getattr_l__mod___features___12___conv_3 => add_77, mul_106, mul_107, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_33', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clweyaatklipsvpnwa7ktr3puvl2jwbkbjhfqrpcmyunfbjv6kwn.py
# Source Nodes: [getattr_l__mod___features___14___conv_1_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___14___conv_1_0 => convolution_40
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bpileobl3jbtroh3cij43snzv2ffwec73rp2sahzogmzcbxnwh.py
# Source Nodes: [getattr_l__mod___features___14___conv_1_1, getattr_l__mod___features___14___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___14___conv_1_1 => add_89, mul_121, mul_122, sub_40
# getattr_l__mod___features___14___conv_1_2 => clamp_max_27, clamp_min_27
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwozzg6b2enbfxhe6xvlugcrwrbjzpmhkvv4mr4uc4cywsq55z7.py
# Source Nodes: [getattr_l__mod___features___14___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___14___conv_2 => convolution_41
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


# kernel path: /tmp/torchinductor_youkaichao/o6/co66gbfss43bjsj7ksqzuvou6ivwpoexv3axuuc3adkpvx4sxjmj.py
# Source Nodes: [getattr_l__mod___features___14___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___14___conv_3 => add_91, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vwmb3curocblgtqwezztd7h3cvlf5fq6rn2yl3je35mv6qfo5a.py
# Source Nodes: [getattr_l__mod___features___15___conv_0_0], Original ATen: [aten.convolution]
# getattr_l__mod___features___15___conv_0_0 => convolution_42
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jekl5sdsw36gvffqmbzvrlrdrycuv67ejbshzys4vrulqfqjcj.py
# Source Nodes: [getattr_l__mod___features___15___conv_0_1, getattr_l__mod___features___15___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# getattr_l__mod___features___15___conv_0_1 => add_93, mul_127, mul_128, sub_42
# getattr_l__mod___features___15___conv_0_2 => clamp_max_28, clamp_min_28
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 <= tmp15
    tmp20 = tmp14 >= tmp17
    tmp21 = tmp19 | tmp20
    tl.store(out_ptr1 + (x2), tmp18, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cecrfpar2slyvrkszvdhcqjwn7cxzzxijqvntbc4fjrp2e3fdctw.py
# Source Nodes: [add_8, getattr_l__mod___features___15___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_8 => add_98
# getattr_l__mod___features___15___conv_3 => add_97, mul_133, mul_134, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_add_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_40', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czewce2spyanumucifs2ymr4nmliicptljx57zxkfce4wq7aaq5g.py
# Source Nodes: [getattr_l__mod___features___17___conv_2], Original ATen: [aten.convolution]
# getattr_l__mod___features___17___conv_2 => convolution_50
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pu/cpudy54pehvv2c472x6aybjrcyiv2vostcz3q6vk55pckuzxukwu.py
# Source Nodes: [getattr_l__mod___features___17___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___features___17___conv_3 => add_111, mul_151, mul_152, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumalhcccslnppdsjtqhf3npq4p4uj5gqjvzxzjewebcech5a3ir.py
# Source Nodes: [l__mod___features_18_0], Original ATen: [aten.convolution]
# l__mod___features_18_0 => convolution_51
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2a/c2ac3c5s6eeyjmnndvedex3wzlidawqnjyqhcxgnnza5jfzamulq.py
# Source Nodes: [l__mod___features_18_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh_backward]
# l__mod___features_18_1 => add_113, mul_154, mul_155, sub_51
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_backward_44', 'mutated_arg_names': []},
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tmp17 = 6.0
    tmp18 = tmp14 >= tmp17
    tmp19 = tmp16 | tmp18
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/cravf326jqu24yogtx2kx6bmjg3w5hti5tcaujfe552xpya53jro.py
# Source Nodes: [x, x_1, x_2], Original ATen: [aten.hardtanh, aten.mean, aten.view]
# x => clamp_max_34, clamp_min_34
# x_1 => mean
# x_2 => view
triton_per_fused_hardtanh_mean_view_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_mean_view_45', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 6.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
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
    assert_size_stride(primals_10, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (96, ), (1, ))
    assert_size_stride(primals_15, (96, ), (1, ))
    assert_size_stride(primals_16, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_22, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (144, ), (1, ))
    assert_size_stride(primals_24, (144, ), (1, ))
    assert_size_stride(primals_25, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_28, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (144, ), (1, ))
    assert_size_stride(primals_30, (144, ), (1, ))
    assert_size_stride(primals_31, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (144, ), (1, ))
    assert_size_stride(primals_33, (144, ), (1, ))
    assert_size_stride(primals_34, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_54, (32, ), (1, ))
    assert_size_stride(primals_55, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (96, 384, 1, 1), (384, 1, 1, 1))
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
    assert_size_stride(primals_112, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_114, (576, ), (1, ))
    assert_size_stride(primals_115, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (96, ), (1, ))
    assert_size_stride(primals_117, (96, ), (1, ))
    assert_size_stride(primals_118, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_119, (576, ), (1, ))
    assert_size_stride(primals_120, (576, ), (1, ))
    assert_size_stride(primals_121, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (576, ), (1, ))
    assert_size_stride(primals_123, (576, ), (1, ))
    assert_size_stride(primals_124, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_125, (160, ), (1, ))
    assert_size_stride(primals_126, (160, ), (1, ))
    assert_size_stride(primals_127, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_128, (960, ), (1, ))
    assert_size_stride(primals_129, (960, ), (1, ))
    assert_size_stride(primals_130, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (960, ), (1, ))
    assert_size_stride(primals_132, (960, ), (1, ))
    assert_size_stride(primals_133, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_134, (160, ), (1, ))
    assert_size_stride(primals_135, (160, ), (1, ))
    assert_size_stride(primals_136, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (960, ), (1, ))
    assert_size_stride(primals_138, (960, ), (1, ))
    assert_size_stride(primals_139, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (960, ), (1, ))
    assert_size_stride(primals_141, (960, ), (1, ))
    assert_size_stride(primals_142, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_146, (960, ), (1, ))
    assert_size_stride(primals_147, (960, ), (1, ))
    assert_size_stride(primals_148, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (960, ), (1, ))
    assert_size_stride(primals_150, (960, ), (1, ))
    assert_size_stride(primals_151, (320, 960, 1, 1), (960, 1, 1, 1))
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
    assert_size_stride(primals_168, (96, ), (1, ))
    assert_size_stride(primals_169, (96, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (96, ), (1, ))
    assert_size_stride(primals_172, (96, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (144, ), (1, ))
    assert_size_stride(primals_178, (144, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (144, ), (1, ))
    assert_size_stride(primals_181, (144, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (144, ), (1, ))
    assert_size_stride(primals_187, (144, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (144, ), (1, ))
    assert_size_stride(primals_190, (144, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (192, ), (1, ))
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (192, ), (1, ))
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (192, ), (1, ))
    assert_size_stride(primals_208, (192, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (192, ), (1, ))
    assert_size_stride(primals_214, (192, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (192, ), (1, ))
    assert_size_stride(primals_217, (192, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (384, ), (1, ))
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
    assert_size_stride(primals_273, (96, ), (1, ))
    assert_size_stride(primals_274, (96, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (576, ), (1, ))
    assert_size_stride(primals_277, (576, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (576, ), (1, ))
    assert_size_stride(primals_280, (576, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (160, ), (1, ))
    assert_size_stride(primals_283, (160, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (960, ), (1, ))
    assert_size_stride(primals_286, (960, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (960, ), (1, ))
    assert_size_stride(primals_289, (960, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (160, ), (1, ))
    assert_size_stride(primals_292, (160, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (960, ), (1, ))
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (960, ), (1, ))
    assert_size_stride(primals_298, (960, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (160, ), (1, ))
    assert_size_stride(primals_301, (160, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (960, ), (1, ))
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (960, ), (1, ))
    assert_size_stride(primals_307, (960, ), (1, ))
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
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf5 = reinterpret_tensor(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        buf229 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_3.run(buf3, primals_159, primals_160, primals_2, primals_3, buf5, buf229, 1605632, grid=grid(1605632), stream=stream0)
        del primals_3
        # Source Nodes: [getattr_l__mod___features___1___conv_0_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf6, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf7 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___1___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf6, buf7, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf9 = reinterpret_tensor(buf6, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf6  # reuse
        buf228 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___1___conv_0_1, getattr_l__mod___features___1___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_3.run(buf7, primals_162, primals_163, primals_5, primals_6, buf9, buf228, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [getattr_l__mod___features___1___conv_1], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf11 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___1___conv_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf10, buf11, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf10  # reuse
        # Source Nodes: [getattr_l__mod___features___1___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf11, primals_165, primals_166, primals_8, primals_9, buf12, 802816, grid=grid(802816), stream=stream0)
        del primals_9
        # Source Nodes: [getattr_l__mod___features___2___conv_0_0], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 96, 112, 112), (1204224, 12544, 112, 1))
        buf14 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf13, buf14, 384, 12544, grid=grid(384, 12544), stream=stream0)
        buf16 = reinterpret_tensor(buf13, (4, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf13  # reuse
        buf227 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___2___conv_0_1, getattr_l__mod___features___2___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_7.run(buf14, primals_168, primals_169, primals_11, primals_12, buf16, buf227, 4816896, grid=grid(4816896), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_l__mod___features___2___conv_1_0], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf17, (4, 96, 56, 56), (301056, 3136, 56, 1))
        buf18 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf17, buf18, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf20 = reinterpret_tensor(buf17, (4, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf17  # reuse
        buf226 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___2___conv_1_1, getattr_l__mod___features___2___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_9.run(buf18, primals_171, primals_172, primals_14, primals_15, buf20, buf226, 1204224, grid=grid(1204224), stream=stream0)
        del primals_15
        # Source Nodes: [getattr_l__mod___features___2___conv_2], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf22 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf21, buf22, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf21  # reuse
        # Source Nodes: [getattr_l__mod___features___2___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf22, primals_174, primals_175, primals_17, primals_18, buf23, 301056, grid=grid(301056), stream=stream0)
        del primals_18
        # Source Nodes: [getattr_l__mod___features___3___conv_0_0], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf25 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf24, buf25, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf27 = reinterpret_tensor(buf24, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf24  # reuse
        buf225 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___3___conv_0_1, getattr_l__mod___features___3___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_13.run(buf25, primals_177, primals_178, primals_20, primals_21, buf27, buf225, 1806336, grid=grid(1806336), stream=stream0)
        del primals_21
        # Source Nodes: [getattr_l__mod___features___3___conv_1_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf28, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf29 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf28, buf29, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf31 = reinterpret_tensor(buf28, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf28  # reuse
        buf224 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___3___conv_1_1, getattr_l__mod___features___3___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_13.run(buf29, primals_180, primals_181, primals_23, primals_24, buf31, buf224, 1806336, grid=grid(1806336), stream=stream0)
        del primals_24
        # Source Nodes: [getattr_l__mod___features___3___conv_2], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf33 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf32, buf33, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf34 = reinterpret_tensor(buf32, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf32  # reuse
        # Source Nodes: [add, getattr_l__mod___features___3___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf23, buf33, primals_183, primals_184, primals_26, primals_27, buf34, 301056, grid=grid(301056), stream=stream0)
        del primals_27
        # Source Nodes: [getattr_l__mod___features___4___conv_0_0], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf36 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf35, buf36, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf38 = reinterpret_tensor(buf35, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf35  # reuse
        buf223 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___4___conv_0_1, getattr_l__mod___features___4___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_13.run(buf36, primals_186, primals_187, primals_29, primals_30, buf38, buf223, 1806336, grid=grid(1806336), stream=stream0)
        del primals_30
        # Source Nodes: [getattr_l__mod___features___4___conv_1_0], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_31, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf39, (4, 144, 28, 28), (112896, 784, 28, 1))
        buf40 = empty_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf39, buf40, 576, 784, grid=grid(576, 784), stream=stream0)
        buf42 = reinterpret_tensor(buf39, (4, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf39  # reuse
        buf222 = empty_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___4___conv_1_1, getattr_l__mod___features___4___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16.run(buf40, primals_189, primals_190, primals_32, primals_33, buf42, buf222, 451584, grid=grid(451584), stream=stream0)
        del primals_33
        # Source Nodes: [getattr_l__mod___features___4___conv_2], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf44 = empty_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf43, buf44, 128, 784, grid=grid(128, 784), stream=stream0)
        buf45 = reinterpret_tensor(buf43, (4, 32, 28, 28), (25088, 1, 896, 32), 0); del buf43  # reuse
        # Source Nodes: [getattr_l__mod___features___4___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf44, primals_192, primals_193, primals_35, primals_36, buf45, 100352, grid=grid(100352), stream=stream0)
        del primals_36
        # Source Nodes: [getattr_l__mod___features___5___conv_0_0], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 192, 28, 28), (150528, 784, 28, 1))
        buf47 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf46, buf47, 768, 784, grid=grid(768, 784), stream=stream0)
        buf49 = reinterpret_tensor(buf46, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf46  # reuse
        buf221 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___5___conv_0_1, getattr_l__mod___features___5___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf47, primals_195, primals_196, primals_38, primals_39, buf49, buf221, 602112, grid=grid(602112), stream=stream0)
        del primals_39
        # Source Nodes: [getattr_l__mod___features___5___conv_1_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf50, (4, 192, 28, 28), (150528, 784, 28, 1))
        buf51 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf50, buf51, 768, 784, grid=grid(768, 784), stream=stream0)
        buf53 = reinterpret_tensor(buf50, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf50  # reuse
        buf220 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___5___conv_1_1, getattr_l__mod___features___5___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf51, primals_198, primals_199, primals_41, primals_42, buf53, buf220, 602112, grid=grid(602112), stream=stream0)
        del primals_42
        # Source Nodes: [getattr_l__mod___features___5___conv_2], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf55 = empty_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf54, buf55, 128, 784, grid=grid(128, 784), stream=stream0)
        buf56 = reinterpret_tensor(buf54, (4, 32, 28, 28), (25088, 1, 896, 32), 0); del buf54  # reuse
        # Source Nodes: [add_1, getattr_l__mod___features___5___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf45, buf55, primals_201, primals_202, primals_44, primals_45, buf56, 100352, grid=grid(100352), stream=stream0)
        del primals_45
        # Source Nodes: [getattr_l__mod___features___6___conv_0_0], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 192, 28, 28), (150528, 784, 28, 1))
        buf58 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf57, buf58, 768, 784, grid=grid(768, 784), stream=stream0)
        buf60 = reinterpret_tensor(buf57, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf57  # reuse
        buf219 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___6___conv_0_1, getattr_l__mod___features___6___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf58, primals_204, primals_205, primals_47, primals_48, buf60, buf219, 602112, grid=grid(602112), stream=stream0)
        del primals_48
        # Source Nodes: [getattr_l__mod___features___6___conv_1_0], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf61, (4, 192, 28, 28), (150528, 784, 28, 1))
        buf62 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf61, buf62, 768, 784, grid=grid(768, 784), stream=stream0)
        buf64 = reinterpret_tensor(buf61, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf61  # reuse
        buf218 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___6___conv_1_1, getattr_l__mod___features___6___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf62, primals_207, primals_208, primals_50, primals_51, buf64, buf218, 602112, grid=grid(602112), stream=stream0)
        del primals_51
        # Source Nodes: [getattr_l__mod___features___6___conv_2], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf66 = empty_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf65, buf66, 128, 784, grid=grid(128, 784), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (4, 32, 28, 28), (25088, 1, 896, 32), 0); del buf65  # reuse
        # Source Nodes: [add_2, getattr_l__mod___features___6___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf56, buf66, primals_210, primals_211, primals_53, primals_54, buf67, 100352, grid=grid(100352), stream=stream0)
        del primals_54
        # Source Nodes: [getattr_l__mod___features___7___conv_0_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 192, 28, 28), (150528, 784, 28, 1))
        buf69 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf68, buf69, 768, 784, grid=grid(768, 784), stream=stream0)
        buf71 = reinterpret_tensor(buf68, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf68  # reuse
        buf217 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___7___conv_0_1, getattr_l__mod___features___7___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf69, primals_213, primals_214, primals_56, primals_57, buf71, buf217, 602112, grid=grid(602112), stream=stream0)
        del primals_57
        # Source Nodes: [getattr_l__mod___features___7___conv_1_0], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_58, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf72, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf73 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf72, buf73, 768, 196, grid=grid(768, 196), stream=stream0)
        buf75 = reinterpret_tensor(buf72, (4, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf72  # reuse
        buf216 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___7___conv_1_1, getattr_l__mod___features___7___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_23.run(buf73, primals_216, primals_217, primals_59, primals_60, buf75, buf216, 150528, grid=grid(150528), stream=stream0)
        del primals_60
        # Source Nodes: [getattr_l__mod___features___7___conv_2], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 64, 14, 14), (12544, 196, 14, 1))
        buf77 = empty_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf76, buf77, 256, 196, grid=grid(256, 196), stream=stream0)
        buf78 = reinterpret_tensor(buf76, (4, 64, 14, 14), (12544, 1, 896, 64), 0); del buf76  # reuse
        # Source Nodes: [getattr_l__mod___features___7___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf77, primals_219, primals_220, primals_62, primals_63, buf78, 50176, grid=grid(50176), stream=stream0)
        del primals_63
        # Source Nodes: [getattr_l__mod___features___8___conv_0_0], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf80 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf79, buf80, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf82 = reinterpret_tensor(buf79, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf79  # reuse
        buf215 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___8___conv_0_1, getattr_l__mod___features___8___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf80, primals_222, primals_223, primals_65, primals_66, buf82, buf215, 301056, grid=grid(301056), stream=stream0)
        del primals_66
        # Source Nodes: [getattr_l__mod___features___8___conv_1_0], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf83, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf84 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf83, buf84, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf86 = reinterpret_tensor(buf83, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf83  # reuse
        buf214 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___8___conv_1_1, getattr_l__mod___features___8___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf84, primals_225, primals_226, primals_68, primals_69, buf86, buf214, 301056, grid=grid(301056), stream=stream0)
        del primals_69
        # Source Nodes: [getattr_l__mod___features___8___conv_2], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 64, 14, 14), (12544, 196, 14, 1))
        buf88 = empty_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf87, buf88, 256, 196, grid=grid(256, 196), stream=stream0)
        buf89 = reinterpret_tensor(buf87, (4, 64, 14, 14), (12544, 1, 896, 64), 0); del buf87  # reuse
        # Source Nodes: [add_3, getattr_l__mod___features___8___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf78, buf88, primals_228, primals_229, primals_71, primals_72, buf89, 50176, grid=grid(50176), stream=stream0)
        del primals_72
        # Source Nodes: [getattr_l__mod___features___9___conv_0_0], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf91 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf90, buf91, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf93 = reinterpret_tensor(buf90, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf90  # reuse
        buf213 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___9___conv_0_1, getattr_l__mod___features___9___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf91, primals_231, primals_232, primals_74, primals_75, buf93, buf213, 301056, grid=grid(301056), stream=stream0)
        del primals_75
        # Source Nodes: [getattr_l__mod___features___9___conv_1_0], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf94, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf95 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf94, buf95, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf97 = reinterpret_tensor(buf94, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf94  # reuse
        buf212 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___9___conv_1_1, getattr_l__mod___features___9___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf95, primals_234, primals_235, primals_77, primals_78, buf97, buf212, 301056, grid=grid(301056), stream=stream0)
        del primals_78
        # Source Nodes: [getattr_l__mod___features___9___conv_2], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 64, 14, 14), (12544, 196, 14, 1))
        buf99 = empty_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf98, buf99, 256, 196, grid=grid(256, 196), stream=stream0)
        buf100 = reinterpret_tensor(buf98, (4, 64, 14, 14), (12544, 1, 896, 64), 0); del buf98  # reuse
        # Source Nodes: [add_4, getattr_l__mod___features___9___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf89, buf99, primals_237, primals_238, primals_80, primals_81, buf100, 50176, grid=grid(50176), stream=stream0)
        del primals_81
        # Source Nodes: [getattr_l__mod___features___10___conv_0_0], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf102 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf101, buf102, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf104 = reinterpret_tensor(buf101, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf101  # reuse
        buf211 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___10___conv_0_1, getattr_l__mod___features___10___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf102, primals_240, primals_241, primals_83, primals_84, buf104, buf211, 301056, grid=grid(301056), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_l__mod___features___10___conv_1_0], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf105, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf106 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf105, buf106, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf108 = reinterpret_tensor(buf105, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf105  # reuse
        buf210 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___10___conv_1_1, getattr_l__mod___features___10___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf106, primals_243, primals_244, primals_86, primals_87, buf108, buf210, 301056, grid=grid(301056), stream=stream0)
        del primals_87
        # Source Nodes: [getattr_l__mod___features___10___conv_2], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 14, 14), (12544, 196, 14, 1))
        buf110 = empty_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf109, buf110, 256, 196, grid=grid(256, 196), stream=stream0)
        buf111 = reinterpret_tensor(buf109, (4, 64, 14, 14), (12544, 1, 896, 64), 0); del buf109  # reuse
        # Source Nodes: [add_5, getattr_l__mod___features___10___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf100, buf110, primals_246, primals_247, primals_89, primals_90, buf111, 50176, grid=grid(50176), stream=stream0)
        del primals_90
        # Source Nodes: [getattr_l__mod___features___11___conv_0_0], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf113 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf112, buf113, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf115 = reinterpret_tensor(buf112, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf112  # reuse
        buf209 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___11___conv_0_1, getattr_l__mod___features___11___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf113, primals_249, primals_250, primals_92, primals_93, buf115, buf209, 301056, grid=grid(301056), stream=stream0)
        del primals_93
        # Source Nodes: [getattr_l__mod___features___11___conv_1_0], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf116, (4, 384, 14, 14), (75264, 196, 14, 1))
        buf117 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf116, buf117, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf119 = reinterpret_tensor(buf116, (4, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf116  # reuse
        buf208 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___11___conv_1_1, getattr_l__mod___features___11___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_27.run(buf117, primals_252, primals_253, primals_95, primals_96, buf119, buf208, 301056, grid=grid(301056), stream=stream0)
        del primals_96
        # Source Nodes: [getattr_l__mod___features___11___conv_2], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 96, 14, 14), (18816, 196, 14, 1))
        buf121 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf120, buf121, 384, 196, grid=grid(384, 196), stream=stream0)
        buf122 = reinterpret_tensor(buf120, (4, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf120  # reuse
        # Source Nodes: [getattr_l__mod___features___11___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf121, primals_255, primals_256, primals_98, primals_99, buf122, 75264, grid=grid(75264), stream=stream0)
        del primals_99
        # Source Nodes: [getattr_l__mod___features___12___conv_0_0], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf124 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf123, buf124, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf126 = reinterpret_tensor(buf123, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf123  # reuse
        buf207 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___12___conv_0_1, getattr_l__mod___features___12___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32.run(buf124, primals_258, primals_259, primals_101, primals_102, buf126, buf207, 451584, grid=grid(451584), stream=stream0)
        del primals_102
        # Source Nodes: [getattr_l__mod___features___12___conv_1_0], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf127, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf128 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf127, buf128, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf130 = reinterpret_tensor(buf127, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf127  # reuse
        buf206 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___12___conv_1_1, getattr_l__mod___features___12___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32.run(buf128, primals_261, primals_262, primals_104, primals_105, buf130, buf206, 451584, grid=grid(451584), stream=stream0)
        del primals_105
        # Source Nodes: [getattr_l__mod___features___12___conv_2], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 96, 14, 14), (18816, 196, 14, 1))
        buf132 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf131, buf132, 384, 196, grid=grid(384, 196), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (4, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf131  # reuse
        # Source Nodes: [add_6, getattr_l__mod___features___12___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_33.run(buf122, buf132, primals_264, primals_265, primals_107, primals_108, buf133, 75264, grid=grid(75264), stream=stream0)
        del primals_108
        # Source Nodes: [getattr_l__mod___features___13___conv_0_0], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf135 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf134, buf135, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf137 = reinterpret_tensor(buf134, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf134  # reuse
        buf205 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___13___conv_0_1, getattr_l__mod___features___13___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32.run(buf135, primals_267, primals_268, primals_110, primals_111, buf137, buf205, 451584, grid=grid(451584), stream=stream0)
        del primals_111
        # Source Nodes: [getattr_l__mod___features___13___conv_1_0], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf138, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf139 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf138, buf139, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf141 = reinterpret_tensor(buf138, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf138  # reuse
        buf204 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___13___conv_1_1, getattr_l__mod___features___13___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32.run(buf139, primals_270, primals_271, primals_113, primals_114, buf141, buf204, 451584, grid=grid(451584), stream=stream0)
        del primals_114
        # Source Nodes: [getattr_l__mod___features___13___conv_2], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 96, 14, 14), (18816, 196, 14, 1))
        buf143 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf142, buf143, 384, 196, grid=grid(384, 196), stream=stream0)
        buf144 = reinterpret_tensor(buf142, (4, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf142  # reuse
        # Source Nodes: [add_7, getattr_l__mod___features___13___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_33.run(buf133, buf143, primals_273, primals_274, primals_116, primals_117, buf144, 75264, grid=grid(75264), stream=stream0)
        del primals_117
        # Source Nodes: [getattr_l__mod___features___14___conv_0_0], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 576, 14, 14), (112896, 196, 14, 1))
        buf146 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf145, buf146, 2304, 196, grid=grid(2304, 196), stream=stream0)
        buf148 = reinterpret_tensor(buf145, (4, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf145  # reuse
        buf203 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___14___conv_0_1, getattr_l__mod___features___14___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_32.run(buf146, primals_276, primals_277, primals_119, primals_120, buf148, buf203, 451584, grid=grid(451584), stream=stream0)
        del primals_120
        # Source Nodes: [getattr_l__mod___features___14___conv_1_0], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_121, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf149, (4, 576, 7, 7), (28224, 49, 7, 1))
        buf150 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf149, buf150, 2304, 49, grid=grid(2304, 49), stream=stream0)
        buf152 = reinterpret_tensor(buf149, (4, 576, 7, 7), (28224, 1, 4032, 576), 0); del buf149  # reuse
        buf202 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___14___conv_1_1, getattr_l__mod___features___14___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_35.run(buf150, primals_279, primals_280, primals_122, primals_123, buf152, buf202, 112896, grid=grid(112896), stream=stream0)
        del primals_123
        # Source Nodes: [getattr_l__mod___features___14___conv_2], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf154 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf153, buf154, 640, 49, grid=grid(640, 49), stream=stream0)
        buf155 = reinterpret_tensor(buf153, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf153  # reuse
        # Source Nodes: [getattr_l__mod___features___14___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf154, primals_282, primals_283, primals_125, primals_126, buf155, 31360, grid=grid(31360), stream=stream0)
        del primals_126
        # Source Nodes: [getattr_l__mod___features___15___conv_0_0], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf157 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf156, buf157, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf159 = reinterpret_tensor(buf156, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf156  # reuse
        buf201 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___15___conv_0_1, getattr_l__mod___features___15___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf157, primals_285, primals_286, primals_128, primals_129, buf159, buf201, 188160, grid=grid(188160), stream=stream0)
        del primals_129
        # Source Nodes: [getattr_l__mod___features___15___conv_1_0], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf160, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf161 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf160, buf161, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf163 = reinterpret_tensor(buf160, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf160  # reuse
        buf200 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___15___conv_1_1, getattr_l__mod___features___15___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf161, primals_288, primals_289, primals_131, primals_132, buf163, buf200, 188160, grid=grid(188160), stream=stream0)
        del primals_132
        # Source Nodes: [getattr_l__mod___features___15___conv_2], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf165 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf164, buf165, 640, 49, grid=grid(640, 49), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf164  # reuse
        # Source Nodes: [add_8, getattr_l__mod___features___15___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf155, buf165, primals_291, primals_292, primals_134, primals_135, buf166, 31360, grid=grid(31360), stream=stream0)
        del primals_135
        # Source Nodes: [getattr_l__mod___features___16___conv_0_0], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf168 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___16___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf167, buf168, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf170 = reinterpret_tensor(buf167, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf167  # reuse
        buf199 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___16___conv_0_1, getattr_l__mod___features___16___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf168, primals_294, primals_295, primals_137, primals_138, buf170, buf199, 188160, grid=grid(188160), stream=stream0)
        del primals_138
        # Source Nodes: [getattr_l__mod___features___16___conv_1_0], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf171, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf172 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___16___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf171, buf172, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf174 = reinterpret_tensor(buf171, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf171  # reuse
        buf198 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___16___conv_1_1, getattr_l__mod___features___16___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf172, primals_297, primals_298, primals_140, primals_141, buf174, buf198, 188160, grid=grid(188160), stream=stream0)
        del primals_141
        # Source Nodes: [getattr_l__mod___features___16___conv_2], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 160, 7, 7), (7840, 49, 7, 1))
        buf176 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___16___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf175, buf176, 640, 49, grid=grid(640, 49), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (4, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf175  # reuse
        # Source Nodes: [add_9, getattr_l__mod___features___16___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf166, buf176, primals_300, primals_301, primals_143, primals_144, buf177, 31360, grid=grid(31360), stream=stream0)
        del primals_144
        # Source Nodes: [getattr_l__mod___features___17___conv_0_0], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf179 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___17___conv_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf178, buf179, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf181 = reinterpret_tensor(buf178, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf178  # reuse
        buf197 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___17___conv_0_1, getattr_l__mod___features___17___conv_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf179, primals_303, primals_304, primals_146, primals_147, buf181, buf197, 188160, grid=grid(188160), stream=stream0)
        del primals_147
        # Source Nodes: [getattr_l__mod___features___17___conv_1_0], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf182, (4, 960, 7, 7), (47040, 49, 7, 1))
        buf183 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___17___conv_1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf182, buf183, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf185 = reinterpret_tensor(buf182, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf182  # reuse
        buf196 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___17___conv_1_1, getattr_l__mod___features___17___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_39.run(buf183, primals_306, primals_307, primals_149, primals_150, buf185, buf196, 188160, grid=grid(188160), stream=stream0)
        del primals_150
        # Source Nodes: [getattr_l__mod___features___17___conv_2], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 320, 7, 7), (15680, 49, 7, 1))
        buf187 = empty_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___17___conv_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf186, buf187, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf188 = reinterpret_tensor(buf186, (4, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf186  # reuse
        # Source Nodes: [getattr_l__mod___features___17___conv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf187, primals_309, primals_310, primals_152, primals_153, buf188, 62720, grid=grid(62720), stream=stream0)
        del primals_153
        # Source Nodes: [l__mod___features_18_0], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 1280, 7, 7), (62720, 49, 7, 1))
        buf190 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_18_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf189, buf190, 5120, 49, grid=grid(5120, 49), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (4, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf189  # reuse
        buf195 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___features_18_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_backward_44.run(buf190, primals_312, primals_313, primals_155, primals_156, buf191, buf195, 250880, grid=grid(250880), stream=stream0)
        del primals_156
        buf192 = empty_strided((4, 1280, 1, 1), (1280, 1, 5120, 5120), device='cuda', dtype=torch.float32)
        buf193 = reinterpret_tensor(buf192, (4, 1280), (1280, 1), 0); del buf192  # reuse
        # Source Nodes: [x, x_1, x_2], Original ATen: [aten.hardtanh, aten.mean, aten.view]
        triton_per_fused_hardtanh_mean_view_45.run(buf193, buf191, 5120, 49, grid=grid(5120), stream=stream0)
        del buf191
        buf194 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf193, reinterpret_tensor(primals_157, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf194)
        del primals_158
        return (buf194, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, buf1, buf3, buf5, buf7, buf9, buf11, buf12, buf14, buf16, buf18, buf20, buf22, buf23, buf25, buf27, buf29, buf31, buf33, buf34, buf36, buf38, buf40, buf42, buf44, buf45, buf47, buf49, buf51, buf53, buf55, buf56, buf58, buf60, buf62, buf64, buf66, buf67, buf69, buf71, buf73, buf75, buf77, buf78, buf80, buf82, buf84, buf86, buf88, buf89, buf91, buf93, buf95, buf97, buf99, buf100, buf102, buf104, buf106, buf108, buf110, buf111, buf113, buf115, buf117, buf119, buf121, buf122, buf124, buf126, buf128, buf130, buf132, buf133, buf135, buf137, buf139, buf141, buf143, buf144, buf146, buf148, buf150, buf152, buf154, buf155, buf157, buf159, buf161, buf163, buf165, buf166, buf168, buf170, buf172, buf174, buf176, buf177, buf179, buf181, buf183, buf185, buf187, buf188, buf190, buf193, reinterpret_tensor(primals_157, (1000, 1280), (1280, 1), 0), buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf227, buf228, buf229, )


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
    primals_10 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_112 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_168 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_273 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    compiled_module_main('mobilenet_v2', benchmark_compiled_module)
