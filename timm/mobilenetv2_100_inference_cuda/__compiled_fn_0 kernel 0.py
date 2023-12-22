
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


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kmxxa52ijq7xnffs3dfpzd4orbfrfo32icpyihfmvdtgu3rsst.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# shortcut => clamp_max, clamp_min
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coet2liwkmp2tlopc2neilt2wszcokbvq2qbpctud5zwhscboet7.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_12 => add_5, mul_7, mul_8, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccn5lc4hn57ia3dv2teiwwgx7u5lcfwnbk6i3akgphzi2jzsdmwd.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_17 => add_7, mul_10, mul_11, sub_3
# x_20 => clamp_max_2, clamp_min_2
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpny5u266u4qbaqwnm27srjwx3hox2uvhrudqnmbgcktcwvt5m3.py
# Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_22 => add_9, mul_13, mul_14, sub_4
# x_25 => clamp_max_3, clamp_min_3
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4u5hy42dcu2z2hgzwgyn2by3rra5xomi4py5rh2ygkmqp2ybfsq.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_28 => add_11, mul_16, mul_17, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 24
    y1 = (yindex // 24)
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgivbfuqytqstgenldeocy3uijstrtvwumjmogbauj3uflukt3c.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_33 => add_13, mul_19, mul_20, sub_6
# x_36 => clamp_max_4, clamp_min_4
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslj64gmbik3oypemkvmodvdlwvc3sdmzrlcdx6frjfbpigkaiyf.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_44 => add_17, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 24
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnniocthis2wmnjlujxy4oqnl5hvhrxd2vckfbb5l744a46jv5ql.py
# Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_55 => add_22, mul_31, mul_32, sub_10
# x_58 => clamp_max_7, clamp_min_7
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2emix2dvnm3c53vyquran4n4w4nlyjtejvchnrvgvkifp3h43lh.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_61 => add_24, mul_34, mul_35, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwg3jtsp5azkpcf3psbwhyiww2ycxwal65xhxyjhscevheknvrbv.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_66 => add_26, mul_37, mul_38, sub_12
# x_69 => clamp_max_8, clamp_min_8
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxys7nhg64kn334jeex37nfieqejwtnfp3zdzmkvzsjqjq4wgeoy.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_31
# x_77 => add_30, mul_43, mul_44, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (32*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsfhevluorjosjwaa5xfope5p37dw64npw6yxoharhkl3yn5jxo.py
# Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_105 => add_42, mul_58, mul_59, sub_19
# x_108 => clamp_max_13, clamp_min_13
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 192
    y1 = (yindex // 192)
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (192*x2) + (37632*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c276u47mzs7n7mkzoppxx6uxwcnmugzfgt56z3lmwaflhncbto7u.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_111 => add_44, mul_61, mul_62, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (12544*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cbl5jftqpc7bee7wei32wjfb5jxypubtzmjckzkx3ajtfazd3t5d.py
# Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_116 => add_46, mul_64, mul_65, sub_21
# x_119 => clamp_max_14, clamp_min_14
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', 'mutated_arg_names': []},
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjv35eoeayaidx33urvlw4xi7fubr6cwhdig3vhzn6hah3woz7s.py
# Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_8 => add_51
# x_127 => add_50, mul_70, mul_71, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chu3vdphpwjgdt2veog24q3q4tdtvveeylzigskkfsyeir7dpkwz.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_178 => add_71, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 96
    y1 = (yindex // 96)
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (18816*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dzl2uwfr72fwh7dee5a4ahgplkelzspqns4fd6yokgnkbspcfc.py
# Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_183 => add_73, mul_100, mul_101, sub_33
# x_186 => clamp_max_22, clamp_min_22
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (576*x2) + (112896*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu24xy2m6srpocgxmmgwe2tpd2zvlwtgymorxvw5rf427ra6o4bj.py
# Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_12 => add_78
# x_194 => add_77, mul_106, mul_107, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 96
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (18816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (96*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevgzw4jygtjs4ax3wvq2nzwsl7q5boqrahxrlxftib77iy3tlej.py
# Source Nodes: [x_222, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_222 => add_89, mul_121, mul_122, sub_40
# x_225 => clamp_max_27, clamp_min_27
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 576
    y1 = (yindex // 576)
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (576*x2) + (28224*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswq22ldqwlhosng5sy4wfjw7rvjlh4erkv52tx4lsbjfngu5trt.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_228 => add_91, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 160
    y1 = (yindex // 160)
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
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s3bxfsyhmhoi7sqd7pgoqvezsocxdxyh77drrg2l5seh363gds.py
# Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_233 => add_93, mul_127, mul_128, sub_42
# x_236 => clamp_max_28, clamp_min_28
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 6.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cessvhvsfwfaursixieait5fa5mltep3pbix6vx5btzrarq4hmkw.py
# Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_15 => add_98
# x_244 => add_97, mul_133, mul_134, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (7840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (160*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/cii22tjz3sdoxt2zsqe36ptm4p3jdgdtr6iqkaxp6xezdyyjztxd.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_278 => add_111, mul_151, mul_152, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
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
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxeyyiad6laz2rh6lkk7kwmemxm22y5kybhtdkqsrrltuy4ocmde.py
# Source Nodes: [x_284, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean]
# x_284 => add_113, mul_154, mul_155, sub_51
# x_288 => clamp_max_34, clamp_min_34
# x_289 => mean
triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
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
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 49.0
    tmp24 = tmp22 / tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (24, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (144, ), (1, ))
    assert_size_stride(arg13_1, (144, ), (1, ))
    assert_size_stride(arg14_1, (144, ), (1, ))
    assert_size_stride(arg15_1, (144, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (144, ), (1, ))
    assert_size_stride(arg19_1, (144, ), (1, ))
    assert_size_stride(arg20_1, (144, ), (1, ))
    assert_size_stride(arg21_1, (144, ), (1, ))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (32, ), (1, ))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (64, ), (1, ))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (96, ), (1, ))
    assert_size_stride(arg65_1, (96, ), (1, ))
    assert_size_stride(arg66_1, (576, ), (1, ))
    assert_size_stride(arg67_1, (576, ), (1, ))
    assert_size_stride(arg68_1, (576, ), (1, ))
    assert_size_stride(arg69_1, (576, ), (1, ))
    assert_size_stride(arg70_1, (96, ), (1, ))
    assert_size_stride(arg71_1, (96, ), (1, ))
    assert_size_stride(arg72_1, (576, ), (1, ))
    assert_size_stride(arg73_1, (576, ), (1, ))
    assert_size_stride(arg74_1, (576, ), (1, ))
    assert_size_stride(arg75_1, (576, ), (1, ))
    assert_size_stride(arg76_1, (96, ), (1, ))
    assert_size_stride(arg77_1, (96, ), (1, ))
    assert_size_stride(arg78_1, (576, ), (1, ))
    assert_size_stride(arg79_1, (576, ), (1, ))
    assert_size_stride(arg80_1, (576, ), (1, ))
    assert_size_stride(arg81_1, (576, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (960, ), (1, ))
    assert_size_stride(arg85_1, (960, ), (1, ))
    assert_size_stride(arg86_1, (960, ), (1, ))
    assert_size_stride(arg87_1, (960, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (960, ), (1, ))
    assert_size_stride(arg91_1, (960, ), (1, ))
    assert_size_stride(arg92_1, (960, ), (1, ))
    assert_size_stride(arg93_1, (960, ), (1, ))
    assert_size_stride(arg94_1, (160, ), (1, ))
    assert_size_stride(arg95_1, (160, ), (1, ))
    assert_size_stride(arg96_1, (960, ), (1, ))
    assert_size_stride(arg97_1, (960, ), (1, ))
    assert_size_stride(arg98_1, (960, ), (1, ))
    assert_size_stride(arg99_1, (960, ), (1, ))
    assert_size_stride(arg100_1, (320, ), (1, ))
    assert_size_stride(arg101_1, (320, ), (1, ))
    assert_size_stride(arg102_1, (1280, ), (1, ))
    assert_size_stride(arg103_1, (1280, ), (1, ))
    assert_size_stride(arg104_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg105_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg107_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg108_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg110_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg111_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg113_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg114_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg115_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg116_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg117_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg119_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg120_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg121_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg122_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg123_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg125_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg126_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg128_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg129_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg130_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg131_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg132_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg134_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg135_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg137_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg138_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg140_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg141_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg143_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg144_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg145_1, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg146_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg147_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg149_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg150_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg152_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg153_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg155_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg156_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg157_1, (1000, ), (1, ))
    assert_size_stride(arg158_1, (32, ), (1, ))
    assert_size_stride(arg159_1, (32, ), (1, ))
    assert_size_stride(arg160_1, (32, ), (1, ))
    assert_size_stride(arg161_1, (32, ), (1, ))
    assert_size_stride(arg162_1, (16, ), (1, ))
    assert_size_stride(arg163_1, (16, ), (1, ))
    assert_size_stride(arg164_1, (96, ), (1, ))
    assert_size_stride(arg165_1, (96, ), (1, ))
    assert_size_stride(arg166_1, (96, ), (1, ))
    assert_size_stride(arg167_1, (96, ), (1, ))
    assert_size_stride(arg168_1, (24, ), (1, ))
    assert_size_stride(arg169_1, (24, ), (1, ))
    assert_size_stride(arg170_1, (144, ), (1, ))
    assert_size_stride(arg171_1, (144, ), (1, ))
    assert_size_stride(arg172_1, (144, ), (1, ))
    assert_size_stride(arg173_1, (144, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (24, ), (1, ))
    assert_size_stride(arg176_1, (144, ), (1, ))
    assert_size_stride(arg177_1, (144, ), (1, ))
    assert_size_stride(arg178_1, (144, ), (1, ))
    assert_size_stride(arg179_1, (144, ), (1, ))
    assert_size_stride(arg180_1, (32, ), (1, ))
    assert_size_stride(arg181_1, (32, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, ), (1, ))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (32, ), (1, ))
    assert_size_stride(arg187_1, (32, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, ), (1, ))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (192, ), (1, ))
    assert_size_stride(arg192_1, (32, ), (1, ))
    assert_size_stride(arg193_1, (32, ), (1, ))
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, ), (1, ))
    assert_size_stride(arg196_1, (192, ), (1, ))
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (64, ), (1, ))
    assert_size_stride(arg199_1, (64, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (64, ), (1, ))
    assert_size_stride(arg205_1, (64, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (64, ), (1, ))
    assert_size_stride(arg211_1, (64, ), (1, ))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (384, ), (1, ))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (64, ), (1, ))
    assert_size_stride(arg217_1, (64, ), (1, ))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (96, ), (1, ))
    assert_size_stride(arg223_1, (96, ), (1, ))
    assert_size_stride(arg224_1, (576, ), (1, ))
    assert_size_stride(arg225_1, (576, ), (1, ))
    assert_size_stride(arg226_1, (576, ), (1, ))
    assert_size_stride(arg227_1, (576, ), (1, ))
    assert_size_stride(arg228_1, (96, ), (1, ))
    assert_size_stride(arg229_1, (96, ), (1, ))
    assert_size_stride(arg230_1, (576, ), (1, ))
    assert_size_stride(arg231_1, (576, ), (1, ))
    assert_size_stride(arg232_1, (576, ), (1, ))
    assert_size_stride(arg233_1, (576, ), (1, ))
    assert_size_stride(arg234_1, (96, ), (1, ))
    assert_size_stride(arg235_1, (96, ), (1, ))
    assert_size_stride(arg236_1, (576, ), (1, ))
    assert_size_stride(arg237_1, (576, ), (1, ))
    assert_size_stride(arg238_1, (576, ), (1, ))
    assert_size_stride(arg239_1, (576, ), (1, ))
    assert_size_stride(arg240_1, (160, ), (1, ))
    assert_size_stride(arg241_1, (160, ), (1, ))
    assert_size_stride(arg242_1, (960, ), (1, ))
    assert_size_stride(arg243_1, (960, ), (1, ))
    assert_size_stride(arg244_1, (960, ), (1, ))
    assert_size_stride(arg245_1, (960, ), (1, ))
    assert_size_stride(arg246_1, (160, ), (1, ))
    assert_size_stride(arg247_1, (160, ), (1, ))
    assert_size_stride(arg248_1, (960, ), (1, ))
    assert_size_stride(arg249_1, (960, ), (1, ))
    assert_size_stride(arg250_1, (960, ), (1, ))
    assert_size_stride(arg251_1, (960, ), (1, ))
    assert_size_stride(arg252_1, (160, ), (1, ))
    assert_size_stride(arg253_1, (160, ), (1, ))
    assert_size_stride(arg254_1, (960, ), (1, ))
    assert_size_stride(arg255_1, (960, ), (1, ))
    assert_size_stride(arg256_1, (960, ), (1, ))
    assert_size_stride(arg257_1, (960, ), (1, ))
    assert_size_stride(arg258_1, (320, ), (1, ))
    assert_size_stride(arg259_1, (320, ), (1, ))
    assert_size_stride(arg260_1, (1280, ), (1, ))
    assert_size_stride(arg261_1, (1280, ), (1, ))
    assert_size_stride(arg262_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg262_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg262_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg104_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg104_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf1
        buf3 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2.run(buf2, arg158_1, arg159_1, arg0_1, arg1_1, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg0_1
        del arg158_1
        del arg159_1
        del arg1_1
        del buf2
        # Source Nodes: [shortcut, x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf4 = extern_kernels.convolution(buf3, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg105_1
        buf5 = buf3; del buf3  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2.run(buf4, arg160_1, arg161_1, arg2_1, arg3_1, buf5, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg160_1
        del arg161_1
        del arg2_1
        del arg3_1
        del buf4
        # Source Nodes: [x_11, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf6 = extern_kernels.convolution(buf5, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg106_1
        del buf5
        buf7 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf6, arg162_1, arg163_1, arg4_1, arg5_1, buf7, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg162_1
        del arg163_1
        del arg4_1
        del arg5_1
        del buf6
        # Source Nodes: [x_12, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg107_1
        del buf7
        buf9 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4.run(buf8, arg164_1, arg165_1, arg6_1, arg7_1, buf9, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del arg164_1
        del arg165_1
        del arg6_1
        del arg7_1
        del buf8
        # Source Nodes: [x_17, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf10 = extern_kernels.convolution(buf9, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf10, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg108_1
        del buf9
        buf11 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf10, arg166_1, arg167_1, arg8_1, arg9_1, buf11, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg166_1
        del arg167_1
        del arg8_1
        del arg9_1
        del buf10
        # Source Nodes: [x_22, x_25, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf12 = extern_kernels.convolution(buf11, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg109_1
        del buf11
        buf13 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf12, arg168_1, arg169_1, arg10_1, arg11_1, buf13, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg168_1
        del arg169_1
        del buf12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg110_1
        buf15 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf14, arg170_1, arg171_1, arg12_1, arg13_1, buf15, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg170_1
        del arg171_1
        del buf14
        # Source Nodes: [x_33, x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf16 = extern_kernels.convolution(buf15, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf16, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg111_1
        buf17 = buf15; del buf15  # reuse
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf16, arg172_1, arg173_1, arg14_1, arg15_1, buf17, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg172_1
        del arg173_1
        del buf16
        # Source Nodes: [x_38, x_41, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf18 = extern_kernels.convolution(buf17, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg112_1
        buf19 = buf13; del buf13  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf19, buf18, arg174_1, arg175_1, arg16_1, arg17_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg16_1
        del arg174_1
        del arg175_1
        del arg17_1
        del buf18
        # Source Nodes: [shortcut_3, x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg113_1
        buf21 = buf17; del buf17  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf20, arg176_1, arg177_1, arg18_1, arg19_1, buf21, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg176_1
        del arg177_1
        del arg18_1
        del arg19_1
        del buf20
        # Source Nodes: [x_50, x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf22 = extern_kernels.convolution(buf21, arg114_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf22, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg114_1
        del buf21
        buf23 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf22, arg178_1, arg179_1, arg20_1, arg21_1, buf23, 1152, 784, grid=grid(1152, 784), stream=stream0)
        del arg178_1
        del arg179_1
        del arg20_1
        del arg21_1
        del buf22
        # Source Nodes: [x_55, x_58, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf24 = extern_kernels.convolution(buf23, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg115_1
        buf25 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf24, arg180_1, arg181_1, arg22_1, arg23_1, buf25, 256, 784, grid=grid(256, 784), stream=stream0)
        del arg180_1
        del arg181_1
        del arg22_1
        del arg23_1
        del buf24
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg116_1
        buf27 = reinterpret_tensor(buf0, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf0  # reuse
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf26, arg182_1, arg183_1, arg24_1, arg25_1, buf27, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg182_1
        del arg183_1
        del arg24_1
        del arg25_1
        del buf26
        # Source Nodes: [x_66, x_69, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf28 = extern_kernels.convolution(buf27, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf28, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg117_1
        buf29 = buf27; del buf27  # reuse
        # Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf28, arg184_1, arg185_1, arg26_1, arg27_1, buf29, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg184_1
        del arg185_1
        del arg26_1
        del arg27_1
        del buf28
        # Source Nodes: [x_71, x_74, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf30 = extern_kernels.convolution(buf29, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg118_1
        buf31 = buf25; del buf25  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf31, buf30, arg186_1, arg187_1, arg28_1, arg29_1, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del arg186_1
        del arg187_1
        del arg28_1
        del arg29_1
        del buf30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg119_1
        buf33 = buf29; del buf29  # reuse
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf32, arg188_1, arg189_1, arg30_1, arg31_1, buf33, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg188_1
        del arg189_1
        del arg30_1
        del arg31_1
        del buf32
        # Source Nodes: [x_83, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf34 = extern_kernels.convolution(buf33, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg120_1
        buf35 = buf33; del buf33  # reuse
        # Source Nodes: [x_88, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf34, arg190_1, arg191_1, arg32_1, arg33_1, buf35, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg190_1
        del arg191_1
        del arg32_1
        del arg33_1
        del buf34
        # Source Nodes: [x_88, x_91, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf36 = extern_kernels.convolution(buf35, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg121_1
        buf37 = buf31; del buf31  # reuse
        # Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf37, buf36, arg192_1, arg193_1, arg34_1, arg35_1, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del arg192_1
        del arg193_1
        del arg34_1
        del arg35_1
        del buf36
        # Source Nodes: [shortcut_6, x_94, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg122_1
        del buf37
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [x_100, x_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf38, arg194_1, arg195_1, arg36_1, arg37_1, buf39, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg194_1
        del arg195_1
        del arg36_1
        del arg37_1
        del buf38
        # Source Nodes: [x_100, x_103, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf40 = extern_kernels.convolution(buf39, arg123_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf40, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg123_1
        del buf39
        buf41 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf40, arg196_1, arg197_1, arg38_1, arg39_1, buf41, 1536, 196, grid=grid(1536, 196), stream=stream0)
        del arg196_1
        del arg197_1
        del arg38_1
        del arg39_1
        del buf40
        # Source Nodes: [x_105, x_108, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf42 = extern_kernels.convolution(buf41, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg124_1
        del buf41
        buf43 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_14.run(buf42, arg198_1, arg199_1, arg40_1, arg41_1, buf43, 512, 196, grid=grid(512, 196), stream=stream0)
        del arg198_1
        del arg199_1
        del arg40_1
        del arg41_1
        del buf42
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg125_1
        buf45 = reinterpret_tensor(buf19, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf19  # reuse
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf44, arg200_1, arg201_1, arg42_1, arg43_1, buf45, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg200_1
        del arg201_1
        del arg42_1
        del arg43_1
        del buf44
        # Source Nodes: [x_116, x_119, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf46 = extern_kernels.convolution(buf45, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf46, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg126_1
        buf47 = buf45; del buf45  # reuse
        # Source Nodes: [x_121, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf46, arg202_1, arg203_1, arg44_1, arg45_1, buf47, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg202_1
        del arg203_1
        del arg44_1
        del arg45_1
        del buf46
        # Source Nodes: [x_121, x_124, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf48 = extern_kernels.convolution(buf47, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg127_1
        buf49 = buf43; del buf43  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf49, buf48, arg204_1, arg205_1, arg46_1, arg47_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg204_1
        del arg205_1
        del arg46_1
        del arg47_1
        del buf48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg128_1
        buf51 = buf47; del buf47  # reuse
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf50, arg206_1, arg207_1, arg48_1, arg49_1, buf51, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg206_1
        del arg207_1
        del arg48_1
        del arg49_1
        del buf50
        # Source Nodes: [x_133, x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf52 = extern_kernels.convolution(buf51, arg129_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf52, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg129_1
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [x_138, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf52, arg208_1, arg209_1, arg50_1, arg51_1, buf53, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg208_1
        del arg209_1
        del arg50_1
        del arg51_1
        del buf52
        # Source Nodes: [x_138, x_141, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf54 = extern_kernels.convolution(buf53, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg130_1
        buf55 = buf49; del buf49  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf55, buf54, arg210_1, arg211_1, arg52_1, arg53_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg210_1
        del arg211_1
        del arg52_1
        del arg53_1
        del buf54
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg131_1
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf56, arg212_1, arg213_1, arg54_1, arg55_1, buf57, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg212_1
        del arg213_1
        del arg54_1
        del arg55_1
        del buf56
        # Source Nodes: [x_150, x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf58 = extern_kernels.convolution(buf57, arg132_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf58, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg132_1
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [x_155, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf58, arg214_1, arg215_1, arg56_1, arg57_1, buf59, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg214_1
        del arg215_1
        del arg56_1
        del arg57_1
        del buf58
        # Source Nodes: [x_155, x_158, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf60 = extern_kernels.convolution(buf59, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg133_1
        buf61 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_10, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf61, buf60, arg216_1, arg217_1, arg58_1, arg59_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg216_1
        del arg217_1
        del arg58_1
        del arg59_1
        del buf60
        # Source Nodes: [shortcut_10, x_161, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg134_1
        del buf61
        buf63 = buf59; del buf59  # reuse
        # Source Nodes: [x_167, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf62, arg218_1, arg219_1, arg60_1, arg61_1, buf63, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg218_1
        del arg219_1
        del arg60_1
        del arg61_1
        del buf62
        # Source Nodes: [x_167, x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf64 = extern_kernels.convolution(buf63, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg135_1
        buf65 = buf63; del buf63  # reuse
        # Source Nodes: [x_172, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf64, arg220_1, arg221_1, arg62_1, arg63_1, buf65, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg220_1
        del arg221_1
        del arg62_1
        del arg63_1
        del buf64
        # Source Nodes: [x_172, x_175, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf66 = extern_kernels.convolution(buf65, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 96, 14, 14), (18816, 196, 14, 1))
        del arg136_1
        del buf65
        buf67 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf66, arg222_1, arg223_1, arg64_1, arg65_1, buf67, 768, 196, grid=grid(768, 196), stream=stream0)
        del arg222_1
        del arg223_1
        del arg64_1
        del arg65_1
        del buf66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg137_1
        buf69 = reinterpret_tensor(buf23, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf23  # reuse
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf68, arg224_1, arg225_1, arg66_1, arg67_1, buf69, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg224_1
        del arg225_1
        del arg66_1
        del arg67_1
        del buf68
        # Source Nodes: [x_183, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf70 = extern_kernels.convolution(buf69, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf70, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg138_1
        buf71 = buf69; del buf69  # reuse
        # Source Nodes: [x_188, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf70, arg226_1, arg227_1, arg68_1, arg69_1, buf71, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg226_1
        del arg227_1
        del arg68_1
        del arg69_1
        del buf70
        # Source Nodes: [x_188, x_191, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf72 = extern_kernels.convolution(buf71, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 96, 14, 14), (18816, 196, 14, 1))
        del arg139_1
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf73, buf72, arg228_1, arg229_1, arg70_1, arg71_1, 1568, 96, grid=grid(1568, 96), stream=stream0)
        del arg228_1
        del arg229_1
        del arg70_1
        del arg71_1
        del buf72
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg140_1
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf74, arg230_1, arg231_1, arg72_1, arg73_1, buf75, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg230_1
        del arg231_1
        del arg72_1
        del arg73_1
        del buf74
        # Source Nodes: [x_200, x_203, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf76 = extern_kernels.convolution(buf75, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf76, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg141_1
        buf77 = buf75; del buf75  # reuse
        # Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf76, arg232_1, arg233_1, arg74_1, arg75_1, buf77, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg232_1
        del arg233_1
        del arg74_1
        del arg75_1
        del buf76
        # Source Nodes: [x_205, x_208, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf78 = extern_kernels.convolution(buf77, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 96, 14, 14), (18816, 196, 14, 1))
        del arg142_1
        buf79 = buf73; del buf73  # reuse
        # Source Nodes: [shortcut_13, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf79, buf78, arg234_1, arg235_1, arg76_1, arg77_1, 1568, 96, grid=grid(1568, 96), stream=stream0)
        del arg234_1
        del arg235_1
        del arg76_1
        del arg77_1
        del buf78
        # Source Nodes: [shortcut_13, x_211, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg143_1
        del buf79
        buf81 = buf77; del buf77  # reuse
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf80, arg236_1, arg237_1, arg78_1, arg79_1, buf81, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg236_1
        del arg237_1
        del arg78_1
        del arg79_1
        del buf80
        # Source Nodes: [x_217, x_220, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf82 = extern_kernels.convolution(buf81, arg144_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf82, (8, 576, 7, 7), (28224, 49, 7, 1))
        del arg144_1
        del buf81
        buf83 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20.run(buf82, arg238_1, arg239_1, arg80_1, arg81_1, buf83, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg238_1
        del arg239_1
        del arg80_1
        del arg81_1
        del buf82
        # Source Nodes: [x_222, x_225, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf84 = extern_kernels.convolution(buf83, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg145_1
        del buf83
        buf85 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf84, arg240_1, arg241_1, arg82_1, arg83_1, buf85, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del arg240_1
        del arg241_1
        del arg82_1
        del arg83_1
        del buf84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg146_1
        buf87 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf86, arg242_1, arg243_1, arg84_1, arg85_1, buf87, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg242_1
        del arg243_1
        del arg84_1
        del arg85_1
        del buf86
        # Source Nodes: [x_233, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf88 = extern_kernels.convolution(buf87, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf88, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg147_1
        buf89 = buf87; del buf87  # reuse
        # Source Nodes: [x_238, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf88, arg244_1, arg245_1, arg86_1, arg87_1, buf89, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg244_1
        del arg245_1
        del arg86_1
        del arg87_1
        del buf88
        # Source Nodes: [x_238, x_241, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf90 = extern_kernels.convolution(buf89, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg148_1
        buf91 = buf85; del buf85  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf91, buf90, arg246_1, arg247_1, arg88_1, arg89_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg246_1
        del arg247_1
        del arg88_1
        del arg89_1
        del buf90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg149_1
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf92, arg248_1, arg249_1, arg90_1, arg91_1, buf93, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg248_1
        del arg249_1
        del arg90_1
        del arg91_1
        del buf92
        # Source Nodes: [x_250, x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf94 = extern_kernels.convolution(buf93, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf94, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg150_1
        buf95 = buf93; del buf93  # reuse
        # Source Nodes: [x_255, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf94, arg250_1, arg251_1, arg92_1, arg93_1, buf95, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg250_1
        del arg251_1
        del arg92_1
        del arg93_1
        del buf94
        # Source Nodes: [x_255, x_258, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf96 = extern_kernels.convolution(buf95, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg151_1
        buf97 = buf91; del buf91  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf97, buf96, arg252_1, arg253_1, arg94_1, arg95_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg252_1
        del arg253_1
        del arg94_1
        del arg95_1
        del buf96
        # Source Nodes: [shortcut_16, x_261, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg152_1
        del buf97
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf98, arg254_1, arg255_1, arg96_1, arg97_1, buf99, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg254_1
        del arg255_1
        del arg96_1
        del arg97_1
        del buf98
        # Source Nodes: [x_267, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf100 = extern_kernels.convolution(buf99, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf100, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg153_1
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [x_272, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf100, arg256_1, arg257_1, arg98_1, arg99_1, buf101, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg256_1
        del arg257_1
        del arg98_1
        del arg99_1
        del buf100
        # Source Nodes: [x_272, x_275, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf102 = extern_kernels.convolution(buf101, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 320, 7, 7), (15680, 49, 7, 1))
        del arg154_1
        del buf101
        buf103 = empty_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf102, arg258_1, arg259_1, arg100_1, arg101_1, buf103, 2560, 49, grid=grid(2560, 49), stream=stream0)
        del arg100_1
        del arg101_1
        del arg258_1
        del arg259_1
        del buf102
        # Source Nodes: [x_278, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 1280, 7, 7), (62720, 49, 7, 1))
        del arg155_1
        del buf103
        buf105 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf106 = reinterpret_tensor(buf105, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf105  # reuse
        # Source Nodes: [x_284, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25.run(buf106, buf104, arg260_1, arg261_1, arg102_1, arg103_1, 10240, 49, grid=grid(10240), stream=stream0)
        del arg102_1
        del arg103_1
        del arg260_1
        del arg261_1
        del buf104
        buf107 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg157_1, reinterpret_tensor(buf106, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg156_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf107)
        del arg156_1
        del arg157_1
        return (buf107, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv2_100', benchmark_compiled_module)
