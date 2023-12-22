
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmtni5pxeiov2pj442gprpdx4myjoozvy47v4amg57hzdlyxpq7.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# shortcut => mul_3, sigmoid
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hssaeq6kwtqbmgjjhbebbbskxqkclzcogigsbfb5e2iwurqqg6.py
# Source Nodes: [x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_12 => clamp_max, clamp_min
# x_7 => add_3, mul_5, mul_6, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhpaie2332om2fdoec3bwrbnyd2qfo6bwyr772366arwrcnjuib.py
# Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_14 => add_5, mul_8, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lw/clw2mldqjclp4pon3znwtj7tzi4rjaodcokzazl54bpkynqyffsg.py
# Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_20 => add_7, mul_11, mul_12, sub_3
# x_24 => mul_13, sigmoid_1
triton_poi_fused__native_batch_norm_legit_no_training_silu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq25l2i6adeev2jpbg322vtms3ivubbwqoph33fnrv6icvt62pwk.py
# Source Nodes: [x_26, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_26 => add_9, mul_15, mul_16, sub_4
# x_31 => clamp_max_1, clamp_min_1
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dk/cdk7s6e5gmg725v2mx55z4vr6gpj2fekvri53tgaqrseaxw7e53n.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_33 => add_11, mul_18, mul_19, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 216
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 27
    y1 = (yindex // 27)
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
    tl.store(out_ptr0 + (y0 + (27*x2) + (84672*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2knwqewlovhyfweey527gnfxj5oyiegxd3rhhetdgdfrqcbjt3.py
# Source Nodes: [x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_39 => add_13, mul_21, mul_22, sub_6
# x_43 => mul_23, sigmoid_2
triton_poi_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 162
    y1 = (yindex // 162)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (162*x2) + (508032*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyvg62trzombrcf63jpxodosjimqj3lfjpkgj7rgm2pqce7mtdv.py
# Source Nodes: [x_45, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# x_45 => add_15, mul_25, mul_26, sub_7
# x_50 => clamp_max_2, clamp_min_2
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 162
    y1 = (yindex // 162)
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
    tl.store(out_ptr0 + (y0 + (162*x2) + (508032*y1)), tmp18, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufic3jeidrksggnzna6oybsp76n2l54saxc3s4r4qeqwrxippcv.py
# Source Nodes: [x_52], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_52 => add_17, mul_28, mul_29, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 38
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvuvoobue2sgo4gzwqfyvsh3xrzyckdhnns3s5ehl3q7vdfmbvp.py
# Source Nodes: [cat_21], Original ATen: [aten.cat]
# cat_21 => cat
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 38
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 27, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (3136*x2) + (119168*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (27*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 38, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (3136*x2) + (119168*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (38*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyewt7lph4j2ch625sav7uwkuiz2nuwywwvpmkdoiz5aos4ispio.py
# Source Nodes: [x_59, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_59 => add_20, mul_31, mul_32, sub_9
# x_63 => mul_33, sigmoid_3
triton_poi_fused__native_batch_norm_legit_no_training_silu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1824
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 228
    y1 = (yindex // 228)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (228*x2) + (715008*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jes6i56kjidw5ag3e5o7pwdldwt2tnmt4omgnqfctzgpejyh24.py
# Source Nodes: [x_65, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_65 => add_22, mul_35, mul_36, sub_10
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1824
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 228
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5c5nhmywzdqigdygodo4gln2sxra4vccdudlg4oa32ox3mpkelw.py
# Source Nodes: [getattr_l__mod___features___3___se_bn, x_se, x_se_1, x_se_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___3___se_bn => add_24, mul_38, mul_39, sub_11
# x_se => mean
# x_se_1 => convolution_11
# x_se_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 19
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cixgj5azhhjyuumx6djbw7ci4l7dfweahlcuphqc5tmxgmovjwua.py
# Source Nodes: [getattr_l__mod___features___3___se_bn, sigmoid, x_70, x_71, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___3___se_bn => add_24, mul_38, mul_39, sub_11
# sigmoid => sigmoid_4
# x_70 => mul_40
# x_71 => clamp_max_3, clamp_min_3
# x_se => mean
# x_se_1 => convolution_11
# x_se_2 => relu
# x_se_3 => convolution_12
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1824
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 228
    y1 = (yindex // 228)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (228*x2) + (178752*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhxo4zeby5qbk72egoyfixvy3fiix4vthyllvx3mxcaovtcyu6n.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_73 => add_26, mul_42, mul_43, sub_12
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 400
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 50
    y1 = (yindex // 50)
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
    tl.store(out_ptr0 + (y0 + (50*x2) + (39200*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csymvgfgpn6yjqquhevin5ycokw5ggxyx7unvaqnkkv4f2vyfdin.py
# Source Nodes: [x_79, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_79 => add_28, mul_45, mul_46, sub_13
# x_83 => mul_47, sigmoid_5
triton_poi_fused__native_batch_norm_legit_no_training_silu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 300
    y1 = (yindex // 300)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (300*x2) + (235200*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gow3igw3ukxzj3n4hewodcou3cpg3d36c3arw647kwguv2amgj.py
# Source Nodes: [x_85, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_85 => add_30, mul_49, mul_50, sub_14
# x_se_4 => mean_1
triton_per_fused__native_batch_norm_legit_no_training_mean_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2400
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 300
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2bhs6whwgaplsvknf7snmbsqnkziamepgannfkgrefo3lykano.py
# Source Nodes: [getattr_l__mod___features___4___se_bn, x_se_4, x_se_5, x_se_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___4___se_bn => add_32, mul_52, mul_53, sub_15
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvkdhjxndeajexex22fct7h5j3tx7enwzbzklcunmwxwsfnqzef.py
# Source Nodes: [getattr_l__mod___features___4___se_bn, sigmoid_1, x_90, x_91, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___4___se_bn => add_32, mul_52, mul_53, sub_15
# sigmoid_1 => sigmoid_6
# x_90 => mul_54
# x_91 => clamp_max_4, clamp_min_4
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
# x_se_7 => convolution_17
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 300
    y1 = (yindex // 300)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (300*x2) + (235200*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qrhdsoeqrlogxtolkwhcw2ukfvngrho6hl66hb4nivzrmtdx2q.py
# Source Nodes: [x_93], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_93 => add_34, mul_56, mul_57, sub_16
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 382592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 61
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvzdqbojgwnaag5i5kqnikwejquso5d3gkz5gk3gxtf7zwbidjl.py
# Source Nodes: [cat_20], Original ATen: [aten.cat]
# cat_20 => cat_1
triton_poi_fused_cat_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 61
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 50, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (784*x2) + (47824*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (50*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 61, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (784*x2) + (47824*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (61*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/cey75c2jel5nq6xyisdctulachsh2g66yjhaqkxi4la3plsus65h.py
# Source Nodes: [x_100, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_100 => add_37, mul_59, mul_60, sub_17
# x_104 => mul_61, sigmoid_7
triton_poi_fused__native_batch_norm_legit_no_training_silu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2928
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 366
    y1 = (yindex // 366)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (366*x2) + (286944*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/cso5xbhzjtakjluwgoggbk2stgb3wywi4hszywscpyg7ecjsp3mv.py
# Source Nodes: [x_106, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_106 => add_39, mul_63, mul_64, sub_18
# x_se_8 => mean_2
triton_per_fused__native_batch_norm_legit_no_training_mean_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2928
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 366
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3ew6frhz6vmd3k6xqqxmt4w6kdrhxmm4mw6f6nvqi3hgvlybrg3.py
# Source Nodes: [getattr_l__mod___features___5___se_bn, x_se_10, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___5___se_bn => add_41, mul_66, mul_67, sub_19
# x_se_10 => relu_2
# x_se_8 => mean_2
# x_se_9 => convolution_21
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covvejada52etwx35dhi5dtn7djon3ny5ckjcs2eroerphhpylho.py
# Source Nodes: [getattr_l__mod___features___5___se_bn, sigmoid_2, x_111, x_112, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___5___se_bn => add_41, mul_66, mul_67, sub_19
# sigmoid_2 => sigmoid_8
# x_111 => mul_68
# x_112 => clamp_max_5, clamp_min_5
# x_se_10 => relu_2
# x_se_11 => convolution_22
# x_se_8 => mean_2
# x_se_9 => convolution_21
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2928
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 366
    y1 = (yindex // 366)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (366*x2) + (71736*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnoawx6dlainhpj44tslendjvbpj7wrn22o4u3v3nf7c5a4ckdmr.py
# Source Nodes: [x_114], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_114 => add_43, mul_70, mul_71, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
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
    tl.store(out_ptr0 + (y0 + (72*x2) + (14112*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmmmliwe6gwqvp3m23tfpde2srn4xj3kf3sjgqt575jlx3qbgaf.py
# Source Nodes: [x_120, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_120 => add_45, mul_73, mul_74, sub_21
# x_124 => mul_75, sigmoid_9
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (432*x2) + (84672*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqui5evtla5i4icnzxxsnnruafolmcn5laxozz47e6iqacf3mtwo.py
# Source Nodes: [x_126, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_126 => add_47, mul_77, mul_78, sub_22
# x_se_12 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_mean_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qlbkk42uphbxfhxpa47xrahumlsuaprzjorbn3blnf5c3glpla.py
# Source Nodes: [getattr_l__mod___features___6___se_bn, x_se_12, x_se_13, x_se_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___6___se_bn => add_49, mul_80, mul_81, sub_23
# x_se_12 => mean_3
# x_se_13 => convolution_26
# x_se_14 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3oas6xr3m2jix5zhyiweop4wwrqzutpwx52xbssgc3nplzbqeb4.py
# Source Nodes: [getattr_l__mod___features___6___se_bn, sigmoid_3, x_131, x_132, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___6___se_bn => add_49, mul_80, mul_81, sub_23
# sigmoid_3 => sigmoid_10
# x_131 => mul_82
# x_132 => clamp_max_6, clamp_min_6
# x_se_12 => mean_3
# x_se_13 => convolution_26
# x_se_14 => relu_3
# x_se_15 => convolution_27
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (432*x2) + (84672*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/cto6pig35p5wwca7awnktpxqscaifezcc22jg3oi2qdi327dqp3w.py
# Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_134 => add_51, mul_84, mul_85, sub_24
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 84
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgu5jzxjmjyxb5va5pmurh4drvuxvzgvtfwyyba4aum6qnlodztb.py
# Source Nodes: [cat_19], Original ATen: [aten.cat]
# cat_19 => cat_2
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 84
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 72, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (196*x2) + (16464*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (72*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 84, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (196*x2) + (16464*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (84*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vsxtxlqwlk3f7ah5syfh6mfew7fsevi7jvrsjszhkf4oqsjue2.py
# Source Nodes: [x_141, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_141 => add_54, mul_87, mul_88, sub_25
# x_145 => mul_89, sigmoid_11
triton_poi_fused__native_batch_norm_legit_no_training_silu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4032
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 504
    y1 = (yindex // 504)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (504*x2) + (98784*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cda2epyma3bb7lq42q7xn5lfibt4n3lazi7e3i3dkjs5uqmvxbqp.py
# Source Nodes: [x_147, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_147 => add_56, mul_91, mul_92, sub_26
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 504
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmtbifuuimrbznbrltunpvyp7cv752njrnfxcpxlabu7ofxqgdn.py
# Source Nodes: [getattr_l__mod___features___7___se_bn, x_se_16, x_se_17, x_se_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___7___se_bn => add_58, mul_94, mul_95, sub_27
# x_se_16 => mean_4
# x_se_17 => convolution_31
# x_se_18 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 42
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjui5ce4puuogmvrttpqcdvsekggmgbjgpsvjbrbehjfrzhd2skj.py
# Source Nodes: [getattr_l__mod___features___7___se_bn, sigmoid_4, x_152, x_153, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___7___se_bn => add_58, mul_94, mul_95, sub_27
# sigmoid_4 => sigmoid_12
# x_152 => mul_96
# x_153 => clamp_max_7, clamp_min_7
# x_se_16 => mean_4
# x_se_17 => convolution_31
# x_se_18 => relu_4
# x_se_19 => convolution_32
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4032
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 504
    y1 = (yindex // 504)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (504*x2) + (98784*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pmzpzxg5n2fb3mzrmop4wtpiydje4ipvsmesvqfxjhq426i66t.py
# Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_155 => add_60, mul_98, mul_99, sub_28
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 148960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 95
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tllo47tfk4xzriaqbpic7onvvop3d5l3uagisecmqeq4zc4tyt.py
# Source Nodes: [cat_18], Original ATen: [aten.cat]
# cat_18 => cat_3
triton_poi_fused_cat_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 95
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 84, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (196*x2) + (18620*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (84*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 95, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (196*x2) + (18620*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (95*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvr2ver63lz64bgi4arp6yz3tzqqqpbvijk67tdqan23laviwap.py
# Source Nodes: [x_162, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_162 => add_63, mul_101, mul_102, sub_29
# x_166 => mul_103, sigmoid_13
triton_poi_fused__native_batch_norm_legit_no_training_silu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4560
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 570
    y1 = (yindex // 570)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (570*x2) + (111720*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnshttr6lazcsa65hnlo7vesuorq7xcakcdbthk45xjtzbjfprln.py
# Source Nodes: [x_168, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_168 => add_65, mul_105, mul_106, sub_30
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_mean_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4560
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 570
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chs6paanekemo76catuen7jgd3tpcsgp2gdzh7g5vfg7gnxfhrbd.py
# Source Nodes: [getattr_l__mod___features___8___se_bn, x_se_20, x_se_21, x_se_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___8___se_bn => add_67, mul_108, mul_109, sub_31
# x_se_20 => mean_5
# x_se_21 => convolution_36
# x_se_22 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 47
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2qlxfiyr7vchj2xc2aix4oc53ntlfbt6xnazvtdtqrwjydnyvc.py
# Source Nodes: [getattr_l__mod___features___8___se_bn, sigmoid_5, x_173, x_174, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___8___se_bn => add_67, mul_108, mul_109, sub_31
# sigmoid_5 => sigmoid_14
# x_173 => mul_110
# x_174 => clamp_max_8, clamp_min_8
# x_se_20 => mean_5
# x_se_21 => convolution_36
# x_se_22 => relu_5
# x_se_23 => convolution_37
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4560
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 570
    y1 = (yindex // 570)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (570*x2) + (111720*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdowco3xjhhevgdrjnegbmjjkhm46gzhgjocgmx55dzsv2ejfsel.py
# Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_176 => add_69, mul_112, mul_113, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 166208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 106
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtke633p4fxnsaucqq46e5yge25vcxw3fblxc2b3sbikij2the5.py
# Source Nodes: [cat_17], Original ATen: [aten.cat]
# cat_17 => cat_4
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 106
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 95, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (196*x2) + (20776*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (95*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 106, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (196*x2) + (20776*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (106*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssdymqccoww7ciuu5ij2ban6pc4hosi4pmfo5soc4bwhhmpohjm.py
# Source Nodes: [x_183, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_183 => add_72, mul_115, mul_116, sub_33
# x_187 => mul_117, sigmoid_15
triton_poi_fused__native_batch_norm_legit_no_training_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5088
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 636
    y1 = (yindex // 636)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (636*x2) + (124656*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pen6fzusvfkd27cjwirmm6xzm2nwhanfxsjihpwok5qzzqf3l2.py
# Source Nodes: [x_189, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_189 => add_74, mul_119, mul_120, sub_34
# x_se_24 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_mean_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_47', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 636
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxn3l63a7f32usczt47cg3yvjwqvyz5wigyz5s3oznizfqen4vh.py
# Source Nodes: [getattr_l__mod___features___9___se_bn, x_se_24, x_se_25, x_se_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___9___se_bn => add_76, mul_122, mul_123, sub_35
# x_se_24 => mean_6
# x_se_25 => convolution_41
# x_se_26 => relu_6
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 53
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccf253lafie4p6w65k5nou5wn6gi6hh2wkouhzcl7kgzqs7hd2yo.py
# Source Nodes: [getattr_l__mod___features___9___se_bn, sigmoid_6, x_194, x_195, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___9___se_bn => add_76, mul_122, mul_123, sub_35
# sigmoid_6 => sigmoid_16
# x_194 => mul_124
# x_195 => clamp_max_9, clamp_min_9
# x_se_24 => mean_6
# x_se_25 => convolution_41
# x_se_26 => relu_6
# x_se_27 => convolution_42
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5088
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 636
    y1 = (yindex // 636)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (636*x2) + (124656*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruqeh4srtlpgffcvhye3wbivczxhflpaccfuohpdl3b7wi4dr3t.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_197 => add_78, mul_126, mul_127, sub_36
triton_poi_fused__native_batch_norm_legit_no_training_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 183456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 117
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvzdufzfe3cldmfxd7eaatmku2e4ya3fejt3j5bnsqw2yv53zkf.py
# Source Nodes: [cat_16], Original ATen: [aten.cat]
# cat_16 => cat_5
triton_poi_fused_cat_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 117
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 106, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (196*x2) + (22932*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (106*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 117, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (196*x2) + (22932*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (117*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzwzqndvqzf7dzselb7yhtvdbeo43toaffxf5cwsircfqdn5raj.py
# Source Nodes: [x_204, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_204 => add_81, mul_129, mul_130, sub_37
# x_208 => mul_131, sigmoid_17
triton_poi_fused__native_batch_norm_legit_no_training_silu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5616
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 702
    y1 = (yindex // 702)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (702*x2) + (137592*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgxzc5n6mjkqcnmbwpqp6sf3u4w6quigl6455rhrusjksmimrmz.py
# Source Nodes: [x_210, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_210 => add_83, mul_133, mul_134, sub_38
# x_se_28 => mean_7
triton_per_fused__native_batch_norm_legit_no_training_mean_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_53', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5616
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 702
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45h7coikxq2gn2jukzc4mhwvvyou5al3powoag4lei252bdeqs5.py
# Source Nodes: [getattr_l__mod___features___10___se_bn, x_se_28, x_se_29, x_se_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___10___se_bn => add_85, mul_136, mul_137, sub_39
# x_se_28 => mean_7
# x_se_29 => convolution_46
# x_se_30 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csm7xfn2uvdk5gzw5n37x6h7mahyov35dwnimyrlbpwxywdvgzlq.py
# Source Nodes: [getattr_l__mod___features___10___se_bn, sigmoid_7, x_215, x_216, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___10___se_bn => add_85, mul_136, mul_137, sub_39
# sigmoid_7 => sigmoid_18
# x_215 => mul_138
# x_216 => clamp_max_10, clamp_min_10
# x_se_28 => mean_7
# x_se_29 => convolution_46
# x_se_30 => relu_7
# x_se_31 => convolution_47
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5616
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 702
    y1 = (yindex // 702)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (702*x2) + (137592*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yizdipwfxu5jfew6c5zdrsmg5xpnpvl4thaiq2pfmaotgdrf24.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_218 => add_87, mul_140, mul_141, sub_40
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cck3qhhaopuk2sflyusuq3cfsz3rggjewncjhufillnbk3spagz7.py
# Source Nodes: [cat_15], Original ATen: [aten.cat]
# cat_15 => cat_6
triton_poi_fused_cat_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 117, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (117*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hbmk7iqld43cszabb5bf4kccostqacuvd3bo2uznje7vrzt26m.py
# Source Nodes: [x_225, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_225 => add_90, mul_143, mul_144, sub_41
# x_229 => mul_145, sigmoid_19
triton_poi_fused__native_batch_norm_legit_no_training_silu_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbg2gewiefddxfapvmjqaovhioixi2gdevj2hqho57e3tpvqrc6l.py
# Source Nodes: [x_231, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_231 => add_92, mul_147, mul_148, sub_42
# x_se_32 => mean_8
triton_per_fused__native_batch_norm_legit_no_training_mean_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_59', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxmhahwxzeiqfwegafoxhjlp3k7af3xskvpkaryftketnw72lhi.py
# Source Nodes: [getattr_l__mod___features___11___se_bn, x_se_32, x_se_33, x_se_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___11___se_bn => add_94, mul_150, mul_151, sub_43
# x_se_32 => mean_8
# x_se_33 => convolution_51
# x_se_34 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curmao4qllsg6bmq6grydw7726qqaq4l6lrkw3wxoj7vcddvvwlm.py
# Source Nodes: [getattr_l__mod___features___11___se_bn, sigmoid_8, x_236, x_237, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___11___se_bn => add_94, mul_150, mul_151, sub_43
# sigmoid_8 => sigmoid_20
# x_236 => mul_152
# x_237 => clamp_max_11, clamp_min_11
# x_se_32 => mean_8
# x_se_33 => convolution_51
# x_se_34 => relu_8
# x_se_35 => convolution_52
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceumhhsrbdxtw2rpdmzok54ezgu6f7jrqh345sbglagxuhtuicfl.py
# Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_239 => add_96, mul_154, mul_155, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1120
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 140
    y1 = (yindex // 140)
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
    tl.store(out_ptr0 + (y0 + (140*x2) + (6860*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceaejezoagcsl2g5jsgnmtsvee7xzlfyc672crtvln5xkyx5hmev.py
# Source Nodes: [x_245, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_245 => add_98, mul_157, mul_158, sub_45
# x_249 => mul_159, sigmoid_21
triton_poi_fused__native_batch_norm_legit_no_training_silu_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6720
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 840
    y1 = (yindex // 840)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (840*x2) + (41160*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr6s3fu4fp2xmsoz2xyo4my5ui5itmgqvvncceei3yq76vknvzz.py
# Source Nodes: [x_251, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_251 => add_100, mul_161, mul_162, sub_46
# x_se_36 => mean_9
triton_per_fused__native_batch_norm_legit_no_training_mean_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_64', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 840
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xee5am7bozgx6uafv6wukb6ngxzjczaxgilzfc5zgmelavvrom.py
# Source Nodes: [getattr_l__mod___features___12___se_bn, x_se_36, x_se_37, x_se_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___12___se_bn => add_102, mul_164, mul_165, sub_47
# x_se_36 => mean_9
# x_se_37 => convolution_56
# x_se_38 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 70
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/ccciqqyh4j5262il4i4sig54mk2rmhvj4nrl64ylznb2h62s5nzp.py
# Source Nodes: [getattr_l__mod___features___12___se_bn, sigmoid_9, x_256, x_257, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___12___se_bn => add_102, mul_164, mul_165, sub_47
# sigmoid_9 => sigmoid_22
# x_256 => mul_166
# x_257 => clamp_max_12, clamp_min_12
# x_se_36 => mean_9
# x_se_37 => convolution_56
# x_se_38 => relu_9
# x_se_39 => convolution_57
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6720
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 840
    y1 = (yindex // 840)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (840*x2) + (41160*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfix7n25i27q37yqdfntqlghu5ghk6fp7qrt3k5vo3q7bfoihqhx.py
# Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_259 => add_104, mul_168, mul_169, sub_48
triton_poi_fused__native_batch_norm_legit_no_training_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 151
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipku75ogq453mosky3su76esukkrq75ccifb3g4rywbjpwyy7oz.py
# Source Nodes: [cat_14], Original ATen: [aten.cat]
# cat_14 => cat_7
triton_poi_fused_cat_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 151
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 140, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (49*x2) + (7399*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (140*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 151, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (49*x2) + (7399*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (151*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7ledxp3fa2xklvimlenvcrzue24nahbxq2kcotmiiv7zbhwkvpk.py
# Source Nodes: [x_266, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_266 => add_107, mul_171, mul_172, sub_49
# x_270 => mul_173, sigmoid_23
triton_poi_fused__native_batch_norm_legit_no_training_silu_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7248
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 906
    y1 = (yindex // 906)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (906*x2) + (44394*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnu25wfdhpgwh3dhkzebe7uwrmv7o4nz526xk52hwxcb677mrnh.py
# Source Nodes: [x_272, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_272 => add_109, mul_175, mul_176, sub_50
# x_se_40 => mean_10
triton_per_fused__native_batch_norm_legit_no_training_mean_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_70', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7248
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 906
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tv2qzem54htqrkwemn45n3esmwwyedx3csw6xcqxq2gfpy64iq.py
# Source Nodes: [getattr_l__mod___features___13___se_bn, x_se_40, x_se_41, x_se_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___13___se_bn => add_111, mul_178, mul_179, sub_51
# x_se_40 => mean_10
# x_se_41 => convolution_61
# x_se_42 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 75
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33hwmrwgpbungltffpis33souo5ca4bp6addbbskonminfv53gh.py
# Source Nodes: [getattr_l__mod___features___13___se_bn, sigmoid_10, x_277, x_278, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___13___se_bn => add_111, mul_178, mul_179, sub_51
# sigmoid_10 => sigmoid_24
# x_277 => mul_180
# x_278 => clamp_max_13, clamp_min_13
# x_se_40 => mean_10
# x_se_41 => convolution_61
# x_se_42 => relu_10
# x_se_43 => convolution_62
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7248
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 906
    y1 = (yindex // 906)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (906*x2) + (44394*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuu75xwlzl3om75ikkpblou4oq46lqgkfpmdhwrt4qzfonhiicm.py
# Source Nodes: [x_280], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_280 => add_113, mul_182, mul_183, sub_52
triton_poi_fused__native_batch_norm_legit_no_training_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 162
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyf2bdzgpsr4qw54mibfhbwp3qyoqpj4thqqfmwo7lb74azl5rcp.py
# Source Nodes: [cat_13], Original ATen: [aten.cat]
# cat_13 => cat_8
triton_poi_fused_cat_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 162
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 151, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (49*x2) + (7938*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (151*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 162, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (49*x2) + (7938*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (162*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyu2h5satlvomti5thtn2ztijlwh7gxjlibv5uwotkz34p4eqsc.py
# Source Nodes: [x_287, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_287 => add_116, mul_185, mul_186, sub_53
# x_291 => mul_187, sigmoid_25
triton_poi_fused__native_batch_norm_legit_no_training_silu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7776
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 972
    y1 = (yindex // 972)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (972*x2) + (47628*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevi7both7l7ld7tui3fyk3r63zub3fnftwhedskbodoyko2l3vt.py
# Source Nodes: [x_293, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_293 => add_118, mul_189, mul_190, sub_54
# x_se_44 => mean_11
triton_per_fused__native_batch_norm_legit_no_training_mean_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_76', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7776
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 972
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjseygz3xozi2lzgsff2ecpiqj6zer5jlevifswocxv4o5hzy2ga.py
# Source Nodes: [getattr_l__mod___features___14___se_bn, x_se_44, x_se_45, x_se_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___14___se_bn => add_120, mul_192, mul_193, sub_55
# x_se_44 => mean_11
# x_se_45 => convolution_66
# x_se_46 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 81
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/om/comysvicq4kpehc47c7wzpyd4cqnglt2k2lu6n3rsfairbannis4.py
# Source Nodes: [getattr_l__mod___features___14___se_bn, sigmoid_11, x_298, x_299, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___14___se_bn => add_120, mul_192, mul_193, sub_55
# sigmoid_11 => sigmoid_26
# x_298 => mul_194
# x_299 => clamp_max_14, clamp_min_14
# x_se_44 => mean_11
# x_se_45 => convolution_66
# x_se_46 => relu_11
# x_se_47 => convolution_67
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7776
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 972
    y1 = (yindex // 972)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (972*x2) + (47628*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pt/cptopybe44agkc4xkzwmtnbakbpxx5kzzurmot4pfrygd3fvi62r.py
# Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_301 => add_122, mul_196, mul_197, sub_56
triton_poi_fused__native_batch_norm_legit_no_training_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 174
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxy3gikufcmaia6lx4egakmgdcsx3xgtomxscdxjstjazpzc2cx.py
# Source Nodes: [cat_12], Original ATen: [aten.cat]
# cat_12 => cat_9
triton_poi_fused_cat_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 174
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 162, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (49*x2) + (8526*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (162*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 174, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (49*x2) + (8526*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (174*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvu2okiqf3krhawi4i6fvgddxrqkcwdz7py42br45yzt6bogsgrd.py
# Source Nodes: [x_308, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_308 => add_125, mul_199, mul_200, sub_57
# x_312 => mul_201, sigmoid_27
triton_poi_fused__native_batch_norm_legit_no_training_silu_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8352
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1044
    y1 = (yindex // 1044)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (1044*x2) + (51156*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibui5lpoenc6opbjeedhnl7xqgkyqz3xuc4fjeuzkckdjh6jj6p.py
# Source Nodes: [x_314, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_314 => add_127, mul_203, mul_204, sub_58
# x_se_48 => mean_12
triton_per_fused__native_batch_norm_legit_no_training_mean_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_82', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1044
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7p/c7poqf2gw6wem56wkpjdzwqm24d3zslz2m5xqrmad2euej6fb5mu.py
# Source Nodes: [getattr_l__mod___features___15___se_bn, x_se_48, x_se_49, x_se_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# getattr_l__mod___features___15___se_bn => add_129, mul_206, mul_207, sub_59
# x_se_48 => mean_12
# x_se_49 => convolution_71
# x_se_50 => relu_12
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 87
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexj4dpzmlxvpdxhwuz3dmqciglxzaecorww34pdauhdxgfl7gct.py
# Source Nodes: [getattr_l__mod___features___15___se_bn, sigmoid_12, x_319, x_320, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# getattr_l__mod___features___15___se_bn => add_129, mul_206, mul_207, sub_59
# sigmoid_12 => sigmoid_28
# x_319 => mul_208
# x_320 => clamp_max_15, clamp_min_15
# x_se_48 => mean_12
# x_se_49 => convolution_71
# x_se_50 => relu_12
# x_se_51 => convolution_72
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8352
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1044
    y1 = (yindex // 1044)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (1044*x2) + (51156*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zbjsr6hdtncw3bnmmjwibdaxbn5i74zafnwqo5llb5zmdg54ri.py
# Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_322 => add_131, mul_210, mul_211, sub_60
triton_poi_fused__native_batch_norm_legit_no_training_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 185
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
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5pshzhs5ypeq3sfysywpvlgfdwl7khzpyhsj7fb7gujowr5tbk.py
# Source Nodes: [cat_11], Original ATen: [aten.cat]
# cat_11 => cat_10
triton_poi_fused_cat_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 185
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 174, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (49*x2) + (9065*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (174*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 185, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + (49*x2) + (9065*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x2 + (185*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwakerl3r2te2p5j5xwelnbjlzhabd3drg2fkgo6yjyz4mrk63i.py
# Source Nodes: [x_329, x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_329 => add_134, mul_213, mul_214, sub_61
# x_334 => mul_215, sigmoid_29
# x_335 => mean_13
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_87', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
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
    assert_size_stride(arg10_1, (27, ), (1, ))
    assert_size_stride(arg11_1, (27, ), (1, ))
    assert_size_stride(arg12_1, (162, ), (1, ))
    assert_size_stride(arg13_1, (162, ), (1, ))
    assert_size_stride(arg14_1, (162, ), (1, ))
    assert_size_stride(arg15_1, (162, ), (1, ))
    assert_size_stride(arg16_1, (38, ), (1, ))
    assert_size_stride(arg17_1, (38, ), (1, ))
    assert_size_stride(arg18_1, (228, ), (1, ))
    assert_size_stride(arg19_1, (228, ), (1, ))
    assert_size_stride(arg20_1, (228, ), (1, ))
    assert_size_stride(arg21_1, (228, ), (1, ))
    assert_size_stride(arg22_1, (50, ), (1, ))
    assert_size_stride(arg23_1, (50, ), (1, ))
    assert_size_stride(arg24_1, (300, ), (1, ))
    assert_size_stride(arg25_1, (300, ), (1, ))
    assert_size_stride(arg26_1, (300, ), (1, ))
    assert_size_stride(arg27_1, (300, ), (1, ))
    assert_size_stride(arg28_1, (61, ), (1, ))
    assert_size_stride(arg29_1, (61, ), (1, ))
    assert_size_stride(arg30_1, (366, ), (1, ))
    assert_size_stride(arg31_1, (366, ), (1, ))
    assert_size_stride(arg32_1, (366, ), (1, ))
    assert_size_stride(arg33_1, (366, ), (1, ))
    assert_size_stride(arg34_1, (72, ), (1, ))
    assert_size_stride(arg35_1, (72, ), (1, ))
    assert_size_stride(arg36_1, (432, ), (1, ))
    assert_size_stride(arg37_1, (432, ), (1, ))
    assert_size_stride(arg38_1, (432, ), (1, ))
    assert_size_stride(arg39_1, (432, ), (1, ))
    assert_size_stride(arg40_1, (84, ), (1, ))
    assert_size_stride(arg41_1, (84, ), (1, ))
    assert_size_stride(arg42_1, (504, ), (1, ))
    assert_size_stride(arg43_1, (504, ), (1, ))
    assert_size_stride(arg44_1, (504, ), (1, ))
    assert_size_stride(arg45_1, (504, ), (1, ))
    assert_size_stride(arg46_1, (95, ), (1, ))
    assert_size_stride(arg47_1, (95, ), (1, ))
    assert_size_stride(arg48_1, (570, ), (1, ))
    assert_size_stride(arg49_1, (570, ), (1, ))
    assert_size_stride(arg50_1, (570, ), (1, ))
    assert_size_stride(arg51_1, (570, ), (1, ))
    assert_size_stride(arg52_1, (106, ), (1, ))
    assert_size_stride(arg53_1, (106, ), (1, ))
    assert_size_stride(arg54_1, (636, ), (1, ))
    assert_size_stride(arg55_1, (636, ), (1, ))
    assert_size_stride(arg56_1, (636, ), (1, ))
    assert_size_stride(arg57_1, (636, ), (1, ))
    assert_size_stride(arg58_1, (117, ), (1, ))
    assert_size_stride(arg59_1, (117, ), (1, ))
    assert_size_stride(arg60_1, (702, ), (1, ))
    assert_size_stride(arg61_1, (702, ), (1, ))
    assert_size_stride(arg62_1, (702, ), (1, ))
    assert_size_stride(arg63_1, (702, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (140, ), (1, ))
    assert_size_stride(arg71_1, (140, ), (1, ))
    assert_size_stride(arg72_1, (840, ), (1, ))
    assert_size_stride(arg73_1, (840, ), (1, ))
    assert_size_stride(arg74_1, (840, ), (1, ))
    assert_size_stride(arg75_1, (840, ), (1, ))
    assert_size_stride(arg76_1, (151, ), (1, ))
    assert_size_stride(arg77_1, (151, ), (1, ))
    assert_size_stride(arg78_1, (906, ), (1, ))
    assert_size_stride(arg79_1, (906, ), (1, ))
    assert_size_stride(arg80_1, (906, ), (1, ))
    assert_size_stride(arg81_1, (906, ), (1, ))
    assert_size_stride(arg82_1, (162, ), (1, ))
    assert_size_stride(arg83_1, (162, ), (1, ))
    assert_size_stride(arg84_1, (972, ), (1, ))
    assert_size_stride(arg85_1, (972, ), (1, ))
    assert_size_stride(arg86_1, (972, ), (1, ))
    assert_size_stride(arg87_1, (972, ), (1, ))
    assert_size_stride(arg88_1, (174, ), (1, ))
    assert_size_stride(arg89_1, (174, ), (1, ))
    assert_size_stride(arg90_1, (1044, ), (1, ))
    assert_size_stride(arg91_1, (1044, ), (1, ))
    assert_size_stride(arg92_1, (1044, ), (1, ))
    assert_size_stride(arg93_1, (1044, ), (1, ))
    assert_size_stride(arg94_1, (185, ), (1, ))
    assert_size_stride(arg95_1, (185, ), (1, ))
    assert_size_stride(arg96_1, (1280, ), (1, ))
    assert_size_stride(arg97_1, (1280, ), (1, ))
    assert_size_stride(arg98_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg99_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg101_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg102_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (27, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg104_1, (162, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(arg105_1, (162, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (38, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg107_1, (228, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg108_1, (228, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (19, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg110_1, (19, ), (1, ))
    assert_size_stride(arg111_1, (19, ), (1, ))
    assert_size_stride(arg112_1, (19, ), (1, ))
    assert_size_stride(arg113_1, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(arg114_1, (228, ), (1, ))
    assert_size_stride(arg115_1, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg116_1, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(arg117_1, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg119_1, (25, ), (1, ))
    assert_size_stride(arg120_1, (25, ), (1, ))
    assert_size_stride(arg121_1, (25, ), (1, ))
    assert_size_stride(arg122_1, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(arg123_1, (300, ), (1, ))
    assert_size_stride(arg124_1, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg125_1, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(arg126_1, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg128_1, (30, ), (1, ))
    assert_size_stride(arg129_1, (30, ), (1, ))
    assert_size_stride(arg130_1, (30, ), (1, ))
    assert_size_stride(arg131_1, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(arg132_1, (366, ), (1, ))
    assert_size_stride(arg133_1, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg134_1, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg135_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg137_1, (36, ), (1, ))
    assert_size_stride(arg138_1, (36, ), (1, ))
    assert_size_stride(arg139_1, (36, ), (1, ))
    assert_size_stride(arg140_1, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg141_1, (432, ), (1, ))
    assert_size_stride(arg142_1, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg143_1, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(arg144_1, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg145_1, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg146_1, (42, ), (1, ))
    assert_size_stride(arg147_1, (42, ), (1, ))
    assert_size_stride(arg148_1, (42, ), (1, ))
    assert_size_stride(arg149_1, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(arg150_1, (504, ), (1, ))
    assert_size_stride(arg151_1, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg152_1, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(arg153_1, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg155_1, (47, ), (1, ))
    assert_size_stride(arg156_1, (47, ), (1, ))
    assert_size_stride(arg157_1, (47, ), (1, ))
    assert_size_stride(arg158_1, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(arg159_1, (570, ), (1, ))
    assert_size_stride(arg160_1, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg161_1, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(arg162_1, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg163_1, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg164_1, (53, ), (1, ))
    assert_size_stride(arg165_1, (53, ), (1, ))
    assert_size_stride(arg166_1, (53, ), (1, ))
    assert_size_stride(arg167_1, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(arg168_1, (636, ), (1, ))
    assert_size_stride(arg169_1, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg170_1, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(arg171_1, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg172_1, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg173_1, (58, ), (1, ))
    assert_size_stride(arg174_1, (58, ), (1, ))
    assert_size_stride(arg175_1, (58, ), (1, ))
    assert_size_stride(arg176_1, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg177_1, (702, ), (1, ))
    assert_size_stride(arg178_1, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg179_1, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg180_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg181_1, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg182_1, (64, ), (1, ))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg188_1, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(arg189_1, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg190_1, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg191_1, (70, ), (1, ))
    assert_size_stride(arg192_1, (70, ), (1, ))
    assert_size_stride(arg193_1, (70, ), (1, ))
    assert_size_stride(arg194_1, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(arg195_1, (840, ), (1, ))
    assert_size_stride(arg196_1, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg197_1, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(arg198_1, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg199_1, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg200_1, (75, ), (1, ))
    assert_size_stride(arg201_1, (75, ), (1, ))
    assert_size_stride(arg202_1, (75, ), (1, ))
    assert_size_stride(arg203_1, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(arg204_1, (906, ), (1, ))
    assert_size_stride(arg205_1, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg206_1, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg207_1, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg208_1, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg209_1, (81, ), (1, ))
    assert_size_stride(arg210_1, (81, ), (1, ))
    assert_size_stride(arg211_1, (81, ), (1, ))
    assert_size_stride(arg212_1, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(arg213_1, (972, ), (1, ))
    assert_size_stride(arg214_1, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg215_1, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(arg216_1, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg218_1, (87, ), (1, ))
    assert_size_stride(arg219_1, (87, ), (1, ))
    assert_size_stride(arg220_1, (87, ), (1, ))
    assert_size_stride(arg221_1, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(arg222_1, (1044, ), (1, ))
    assert_size_stride(arg223_1, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg224_1, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(arg225_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg226_1, (1000, ), (1, ))
    assert_size_stride(arg227_1, (32, ), (1, ))
    assert_size_stride(arg228_1, (32, ), (1, ))
    assert_size_stride(arg229_1, (32, ), (1, ))
    assert_size_stride(arg230_1, (32, ), (1, ))
    assert_size_stride(arg231_1, (16, ), (1, ))
    assert_size_stride(arg232_1, (16, ), (1, ))
    assert_size_stride(arg233_1, (96, ), (1, ))
    assert_size_stride(arg234_1, (96, ), (1, ))
    assert_size_stride(arg235_1, (96, ), (1, ))
    assert_size_stride(arg236_1, (96, ), (1, ))
    assert_size_stride(arg237_1, (27, ), (1, ))
    assert_size_stride(arg238_1, (27, ), (1, ))
    assert_size_stride(arg239_1, (162, ), (1, ))
    assert_size_stride(arg240_1, (162, ), (1, ))
    assert_size_stride(arg241_1, (162, ), (1, ))
    assert_size_stride(arg242_1, (162, ), (1, ))
    assert_size_stride(arg243_1, (38, ), (1, ))
    assert_size_stride(arg244_1, (38, ), (1, ))
    assert_size_stride(arg245_1, (228, ), (1, ))
    assert_size_stride(arg246_1, (228, ), (1, ))
    assert_size_stride(arg247_1, (228, ), (1, ))
    assert_size_stride(arg248_1, (228, ), (1, ))
    assert_size_stride(arg249_1, (50, ), (1, ))
    assert_size_stride(arg250_1, (50, ), (1, ))
    assert_size_stride(arg251_1, (300, ), (1, ))
    assert_size_stride(arg252_1, (300, ), (1, ))
    assert_size_stride(arg253_1, (300, ), (1, ))
    assert_size_stride(arg254_1, (300, ), (1, ))
    assert_size_stride(arg255_1, (61, ), (1, ))
    assert_size_stride(arg256_1, (61, ), (1, ))
    assert_size_stride(arg257_1, (366, ), (1, ))
    assert_size_stride(arg258_1, (366, ), (1, ))
    assert_size_stride(arg259_1, (366, ), (1, ))
    assert_size_stride(arg260_1, (366, ), (1, ))
    assert_size_stride(arg261_1, (72, ), (1, ))
    assert_size_stride(arg262_1, (72, ), (1, ))
    assert_size_stride(arg263_1, (432, ), (1, ))
    assert_size_stride(arg264_1, (432, ), (1, ))
    assert_size_stride(arg265_1, (432, ), (1, ))
    assert_size_stride(arg266_1, (432, ), (1, ))
    assert_size_stride(arg267_1, (84, ), (1, ))
    assert_size_stride(arg268_1, (84, ), (1, ))
    assert_size_stride(arg269_1, (504, ), (1, ))
    assert_size_stride(arg270_1, (504, ), (1, ))
    assert_size_stride(arg271_1, (504, ), (1, ))
    assert_size_stride(arg272_1, (504, ), (1, ))
    assert_size_stride(arg273_1, (95, ), (1, ))
    assert_size_stride(arg274_1, (95, ), (1, ))
    assert_size_stride(arg275_1, (570, ), (1, ))
    assert_size_stride(arg276_1, (570, ), (1, ))
    assert_size_stride(arg277_1, (570, ), (1, ))
    assert_size_stride(arg278_1, (570, ), (1, ))
    assert_size_stride(arg279_1, (106, ), (1, ))
    assert_size_stride(arg280_1, (106, ), (1, ))
    assert_size_stride(arg281_1, (636, ), (1, ))
    assert_size_stride(arg282_1, (636, ), (1, ))
    assert_size_stride(arg283_1, (636, ), (1, ))
    assert_size_stride(arg284_1, (636, ), (1, ))
    assert_size_stride(arg285_1, (117, ), (1, ))
    assert_size_stride(arg286_1, (117, ), (1, ))
    assert_size_stride(arg287_1, (702, ), (1, ))
    assert_size_stride(arg288_1, (702, ), (1, ))
    assert_size_stride(arg289_1, (702, ), (1, ))
    assert_size_stride(arg290_1, (702, ), (1, ))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (768, ), (1, ))
    assert_size_stride(arg295_1, (768, ), (1, ))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (140, ), (1, ))
    assert_size_stride(arg298_1, (140, ), (1, ))
    assert_size_stride(arg299_1, (840, ), (1, ))
    assert_size_stride(arg300_1, (840, ), (1, ))
    assert_size_stride(arg301_1, (840, ), (1, ))
    assert_size_stride(arg302_1, (840, ), (1, ))
    assert_size_stride(arg303_1, (151, ), (1, ))
    assert_size_stride(arg304_1, (151, ), (1, ))
    assert_size_stride(arg305_1, (906, ), (1, ))
    assert_size_stride(arg306_1, (906, ), (1, ))
    assert_size_stride(arg307_1, (906, ), (1, ))
    assert_size_stride(arg308_1, (906, ), (1, ))
    assert_size_stride(arg309_1, (162, ), (1, ))
    assert_size_stride(arg310_1, (162, ), (1, ))
    assert_size_stride(arg311_1, (972, ), (1, ))
    assert_size_stride(arg312_1, (972, ), (1, ))
    assert_size_stride(arg313_1, (972, ), (1, ))
    assert_size_stride(arg314_1, (972, ), (1, ))
    assert_size_stride(arg315_1, (174, ), (1, ))
    assert_size_stride(arg316_1, (174, ), (1, ))
    assert_size_stride(arg317_1, (1044, ), (1, ))
    assert_size_stride(arg318_1, (1044, ), (1, ))
    assert_size_stride(arg319_1, (1044, ), (1, ))
    assert_size_stride(arg320_1, (1044, ), (1, ))
    assert_size_stride(arg321_1, (185, ), (1, ))
    assert_size_stride(arg322_1, (185, ), (1, ))
    assert_size_stride(arg323_1, (1280, ), (1, ))
    assert_size_stride(arg324_1, (1280, ), (1, ))
    assert_size_stride(arg325_1, (19, ), (1, ))
    assert_size_stride(arg326_1, (19, ), (1, ))
    assert_size_stride(arg327_1, (), ())
    assert_size_stride(arg328_1, (25, ), (1, ))
    assert_size_stride(arg329_1, (25, ), (1, ))
    assert_size_stride(arg330_1, (), ())
    assert_size_stride(arg331_1, (30, ), (1, ))
    assert_size_stride(arg332_1, (30, ), (1, ))
    assert_size_stride(arg333_1, (), ())
    assert_size_stride(arg334_1, (36, ), (1, ))
    assert_size_stride(arg335_1, (36, ), (1, ))
    assert_size_stride(arg336_1, (), ())
    assert_size_stride(arg337_1, (42, ), (1, ))
    assert_size_stride(arg338_1, (42, ), (1, ))
    assert_size_stride(arg339_1, (), ())
    assert_size_stride(arg340_1, (47, ), (1, ))
    assert_size_stride(arg341_1, (47, ), (1, ))
    assert_size_stride(arg342_1, (), ())
    assert_size_stride(arg343_1, (53, ), (1, ))
    assert_size_stride(arg344_1, (53, ), (1, ))
    assert_size_stride(arg345_1, (), ())
    assert_size_stride(arg346_1, (58, ), (1, ))
    assert_size_stride(arg347_1, (58, ), (1, ))
    assert_size_stride(arg348_1, (), ())
    assert_size_stride(arg349_1, (64, ), (1, ))
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (), ())
    assert_size_stride(arg352_1, (70, ), (1, ))
    assert_size_stride(arg353_1, (70, ), (1, ))
    assert_size_stride(arg354_1, (), ())
    assert_size_stride(arg355_1, (75, ), (1, ))
    assert_size_stride(arg356_1, (75, ), (1, ))
    assert_size_stride(arg357_1, (), ())
    assert_size_stride(arg358_1, (81, ), (1, ))
    assert_size_stride(arg359_1, (81, ), (1, ))
    assert_size_stride(arg360_1, (), ())
    assert_size_stride(arg361_1, (87, ), (1, ))
    assert_size_stride(arg362_1, (87, ), (1, ))
    assert_size_stride(arg363_1, (), ())
    assert_size_stride(arg364_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg364_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg364_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg98_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg98_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg227_1, arg228_1, arg0_1, arg1_1, buf4, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg0_1
        del arg1_1
        del arg227_1
        del arg228_1
        del buf3
        # Source Nodes: [shortcut, x_6], Original ATen: [aten.convolution, aten.silu]
        buf5 = extern_kernels.convolution(buf4, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg99_1
        buf6 = buf4; del buf4  # reuse
        # Source Nodes: [x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf5, arg229_1, arg230_1, arg2_1, arg3_1, buf6, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg229_1
        del arg230_1
        del arg2_1
        del arg3_1
        del buf5
        # Source Nodes: [x_12, x_13, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf7 = extern_kernels.convolution(buf6, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg100_1
        del buf6
        buf8 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf7, arg231_1, arg232_1, arg4_1, arg5_1, buf8, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg231_1
        del arg232_1
        del arg4_1
        del arg5_1
        del buf7
        # Source Nodes: [x_14, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg101_1
        del buf8
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_5.run(buf10, arg233_1, arg234_1, arg6_1, arg7_1, buf11, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del arg233_1
        del arg234_1
        del arg6_1
        del arg7_1
        del buf10
        # Source Nodes: [x_24, x_25], Original ATen: [aten.convolution, aten.silu]
        buf12 = extern_kernels.convolution(buf11, arg102_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg102_1
        del buf11
        buf13 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf12, arg235_1, arg236_1, arg8_1, arg9_1, buf13, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg235_1
        del arg236_1
        del arg8_1
        del arg9_1
        del buf12
        # Source Nodes: [x_26, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf14 = extern_kernels.convolution(buf13, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 27, 56, 56), (84672, 3136, 56, 1))
        del arg103_1
        del buf13
        buf15 = empty_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf14, arg237_1, arg238_1, arg10_1, arg11_1, buf15, 216, 3136, grid=grid(216, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg237_1
        del arg238_1
        del buf14
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 162, 56, 56), (508032, 3136, 56, 1))
        del arg104_1
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf17, arg239_1, arg240_1, arg12_1, arg13_1, buf18, 1296, 3136, grid=grid(1296, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg239_1
        del arg240_1
        del buf17
        # Source Nodes: [x_43, x_44], Original ATen: [aten.convolution, aten.silu]
        buf19 = extern_kernels.convolution(buf18, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=162, bias=None)
        assert_size_stride(buf19, (8, 162, 56, 56), (508032, 3136, 56, 1))
        del arg105_1
        buf20 = buf18; del buf18  # reuse
        # Source Nodes: [x_45, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf19, arg241_1, arg242_1, arg14_1, arg15_1, buf20, 1296, 3136, grid=grid(1296, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg241_1
        del arg242_1
        del buf19
        # Source Nodes: [x_45, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
        buf21 = extern_kernels.convolution(buf20, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 38, 56, 56), (119168, 3136, 56, 1))
        del arg106_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Source Nodes: [x_52], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf22, arg243_1, arg244_1, arg16_1, arg17_1, 953344, grid=grid(953344), stream=stream0)
        del arg16_1
        del arg17_1
        del arg243_1
        del arg244_1
        buf23 = empty_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf22, buf15, buf23, 25088, 38, grid=grid(25088, 38), stream=stream0)
        del buf22
        # Source Nodes: [cat_21, x_58], Original ATen: [aten.cat, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 228, 56, 56), (715008, 3136, 56, 1))
        del arg107_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf25, arg245_1, arg246_1, arg18_1, arg19_1, buf26, 1824, 3136, grid=grid(1824, 3136), stream=stream0)
        del arg18_1
        del arg19_1
        del arg245_1
        del arg246_1
        del buf25
        # Source Nodes: [x_63, x_64], Original ATen: [aten.convolution, aten.silu]
        buf27 = extern_kernels.convolution(buf26, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=228, bias=None)
        assert_size_stride(buf27, (8, 228, 28, 28), (178752, 784, 28, 1))
        del arg108_1
        del buf26
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided((8, 228, 1, 1), (228, 1, 1824, 1824), device='cuda', dtype=torch.float32)
        buf30 = reinterpret_tensor(buf29, (8, 228, 1, 1), (228, 1, 228, 228), 0); del buf29  # reuse
        # Source Nodes: [x_65, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_13.run(buf28, buf30, arg247_1, arg248_1, arg20_1, arg21_1, 1824, 784, grid=grid(1824), stream=stream0)
        del arg20_1
        del arg21_1
        del arg247_1
        del arg248_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf31 = extern_kernels.convolution(buf30, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 19, 1, 1), (19, 1, 1, 1))
        del arg109_1
        del buf30
        buf32 = reinterpret_tensor(buf31, (8, 19, 1, 1), (19, 1, 19, 19), 0); del buf31  # reuse
        # Source Nodes: [getattr_l__mod___features___3___se_bn, x_se, x_se_1, x_se_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_14.run(buf32, arg110_1, arg325_1, arg326_1, arg111_1, arg112_1, 152, grid=grid(152), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg325_1
        del arg326_1
        # Source Nodes: [getattr_l__mod___features___3___se_bn, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf33 = extern_kernels.convolution(buf32, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 228, 1, 1), (228, 1, 1, 1))
        del arg113_1
        del buf32
        buf34 = empty_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___se_bn, sigmoid, x_70, x_71, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_15.run(buf28, buf33, arg114_1, buf34, 1824, 784, grid=grid(1824, 784), stream=stream0)
        del arg114_1
        del buf28
        del buf33
        # Source Nodes: [getattr_l__mod___features___3___se_bn, sigmoid, x_70, x_71, x_72, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf35 = extern_kernels.convolution(buf34, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 50, 28, 28), (39200, 784, 28, 1))
        del arg115_1
        del buf34
        buf36 = empty_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf35, arg249_1, arg250_1, arg22_1, arg23_1, buf36, 400, 784, grid=grid(400, 784), stream=stream0)
        del arg22_1
        del arg23_1
        del arg249_1
        del arg250_1
        del buf35
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 300, 28, 28), (235200, 784, 28, 1))
        del arg116_1
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_17.run(buf38, arg251_1, arg252_1, arg24_1, arg25_1, buf39, 2400, 784, grid=grid(2400, 784), stream=stream0)
        del arg24_1
        del arg251_1
        del arg252_1
        del arg25_1
        del buf38
        # Source Nodes: [x_83, x_84], Original ATen: [aten.convolution, aten.silu]
        buf40 = extern_kernels.convolution(buf39, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=300, bias=None)
        assert_size_stride(buf40, (8, 300, 28, 28), (235200, 784, 28, 1))
        del arg117_1
        buf41 = buf40; del buf40  # reuse
        buf42 = empty_strided((8, 300, 1, 1), (300, 1, 2400, 2400), device='cuda', dtype=torch.float32)
        buf43 = reinterpret_tensor(buf42, (8, 300, 1, 1), (300, 1, 300, 300), 0); del buf42  # reuse
        # Source Nodes: [x_85, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_18.run(buf41, buf43, arg253_1, arg254_1, arg26_1, arg27_1, 2400, 784, grid=grid(2400), stream=stream0)
        del arg253_1
        del arg254_1
        del arg26_1
        del arg27_1
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf44 = extern_kernels.convolution(buf43, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 25, 1, 1), (25, 1, 1, 1))
        del arg118_1
        del buf43
        buf45 = reinterpret_tensor(buf44, (8, 25, 1, 1), (25, 1, 25, 25), 0); del buf44  # reuse
        # Source Nodes: [getattr_l__mod___features___4___se_bn, x_se_4, x_se_5, x_se_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_19.run(buf45, arg119_1, arg328_1, arg329_1, arg120_1, arg121_1, 200, grid=grid(200), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        del arg328_1
        del arg329_1
        # Source Nodes: [getattr_l__mod___features___4___se_bn, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 300, 1, 1), (300, 1, 1, 1))
        del arg122_1
        del buf45
        buf47 = buf39; del buf39  # reuse
        # Source Nodes: [getattr_l__mod___features___4___se_bn, sigmoid_1, x_90, x_91, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_20.run(buf41, buf46, arg123_1, buf47, 2400, 784, grid=grid(2400, 784), stream=stream0)
        del arg123_1
        del buf41
        del buf46
        # Source Nodes: [getattr_l__mod___features___4___se_bn, sigmoid_1, x_90, x_91, x_92, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf48 = extern_kernels.convolution(buf47, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 61, 28, 28), (47824, 784, 28, 1))
        del arg124_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [x_93], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf49, arg255_1, arg256_1, arg28_1, arg29_1, 382592, grid=grid(382592), stream=stream0)
        del arg255_1
        del arg256_1
        del arg28_1
        del arg29_1
        buf50 = empty_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_22.run(buf49, buf36, buf50, 6272, 61, grid=grid(6272, 61), stream=stream0)
        del buf36
        del buf49
        # Source Nodes: [cat_20, x_99], Original ATen: [aten.cat, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 366, 28, 28), (286944, 784, 28, 1))
        del arg125_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        buf53 = empty_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_23.run(buf52, arg257_1, arg258_1, arg30_1, arg31_1, buf53, 2928, 784, grid=grid(2928, 784), stream=stream0)
        del arg257_1
        del arg258_1
        del arg30_1
        del arg31_1
        del buf52
        # Source Nodes: [x_104, x_105], Original ATen: [aten.convolution, aten.silu]
        buf54 = extern_kernels.convolution(buf53, arg126_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=366, bias=None)
        assert_size_stride(buf54, (8, 366, 14, 14), (71736, 196, 14, 1))
        del arg126_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided((8, 366, 1, 1), (366, 1, 2928, 2928), device='cuda', dtype=torch.float32)
        buf57 = reinterpret_tensor(buf56, (8, 366, 1, 1), (366, 1, 366, 366), 0); del buf56  # reuse
        # Source Nodes: [x_106, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_24.run(buf55, buf57, arg259_1, arg260_1, arg32_1, arg33_1, 2928, 196, grid=grid(2928), stream=stream0)
        del arg259_1
        del arg260_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
        buf58 = extern_kernels.convolution(buf57, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 30, 1, 1), (30, 1, 1, 1))
        del arg127_1
        del buf57
        buf59 = reinterpret_tensor(buf58, (8, 30, 1, 1), (30, 1, 30, 30), 0); del buf58  # reuse
        # Source Nodes: [getattr_l__mod___features___5___se_bn, x_se_10, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_25.run(buf59, arg128_1, arg331_1, arg332_1, arg129_1, arg130_1, 240, grid=grid(240), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        del arg331_1
        del arg332_1
        # Source Nodes: [getattr_l__mod___features___5___se_bn, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 366, 1, 1), (366, 1, 1, 1))
        del arg131_1
        del buf59
        buf61 = empty_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___se_bn, sigmoid_2, x_111, x_112, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_26.run(buf55, buf60, arg132_1, buf61, 2928, 196, grid=grid(2928, 196), stream=stream0)
        del arg132_1
        del buf55
        del buf60
        # Source Nodes: [getattr_l__mod___features___5___se_bn, sigmoid_2, x_111, x_112, x_113, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf62 = extern_kernels.convolution(buf61, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg133_1
        del buf61
        buf63 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf62, arg261_1, arg262_1, arg34_1, arg35_1, buf63, 576, 196, grid=grid(576, 196), stream=stream0)
        del arg261_1
        del arg262_1
        del arg34_1
        del arg35_1
        del buf62
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 432, 14, 14), (84672, 196, 14, 1))
        del arg134_1
        buf65 = buf64; del buf64  # reuse
        buf66 = reinterpret_tensor(buf15, (8, 432, 14, 14), (84672, 1, 6048, 432), 0); del buf15  # reuse
        # Source Nodes: [x_120, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf65, arg263_1, arg264_1, arg36_1, arg37_1, buf66, 3456, 196, grid=grid(3456, 196), stream=stream0)
        del arg263_1
        del arg264_1
        del arg36_1
        del arg37_1
        del buf65
        # Source Nodes: [x_124, x_125], Original ATen: [aten.convolution, aten.silu]
        buf67 = extern_kernels.convolution(buf66, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf67, (8, 432, 14, 14), (84672, 196, 14, 1))
        del arg135_1
        buf68 = buf67; del buf67  # reuse
        buf69 = empty_strided((8, 432, 1, 1), (432, 1, 3456, 3456), device='cuda', dtype=torch.float32)
        buf70 = reinterpret_tensor(buf69, (8, 432, 1, 1), (432, 1, 432, 432), 0); del buf69  # reuse
        # Source Nodes: [x_126, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_29.run(buf68, buf70, arg265_1, arg266_1, arg38_1, arg39_1, 3456, 196, grid=grid(3456), stream=stream0)
        del arg265_1
        del arg266_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
        buf71 = extern_kernels.convolution(buf70, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 36, 1, 1), (36, 1, 1, 1))
        del arg136_1
        del buf70
        buf72 = reinterpret_tensor(buf71, (8, 36, 1, 1), (36, 1, 36, 36), 0); del buf71  # reuse
        # Source Nodes: [getattr_l__mod___features___6___se_bn, x_se_12, x_se_13, x_se_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_30.run(buf72, arg137_1, arg334_1, arg335_1, arg138_1, arg139_1, 288, grid=grid(288), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg334_1
        del arg335_1
        # Source Nodes: [getattr_l__mod___features___6___se_bn, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf73 = extern_kernels.convolution(buf72, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 432, 1, 1), (432, 1, 1, 1))
        del arg140_1
        del buf72
        buf74 = buf66; del buf66  # reuse
        # Source Nodes: [getattr_l__mod___features___6___se_bn, sigmoid_3, x_131, x_132, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_31.run(buf68, buf73, arg141_1, buf74, 3456, 196, grid=grid(3456, 196), stream=stream0)
        del arg141_1
        del buf68
        del buf73
        # Source Nodes: [getattr_l__mod___features___6___se_bn, sigmoid_3, x_131, x_132, x_133, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf75 = extern_kernels.convolution(buf74, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 84, 14, 14), (16464, 196, 14, 1))
        del arg142_1
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf76, arg267_1, arg268_1, arg40_1, arg41_1, 131712, grid=grid(131712), stream=stream0)
        del arg267_1
        del arg268_1
        del arg40_1
        del arg41_1
        buf77 = empty_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf76, buf63, buf77, 1568, 84, grid=grid(1568, 84), stream=stream0)
        del buf63
        del buf76
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 504, 14, 14), (98784, 196, 14, 1))
        del arg143_1
        buf79 = buf78; del buf78  # reuse
        buf80 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf79, arg269_1, arg270_1, arg42_1, arg43_1, buf80, 4032, 196, grid=grid(4032, 196), stream=stream0)
        del arg269_1
        del arg270_1
        del arg42_1
        del arg43_1
        del buf79
        # Source Nodes: [x_145, x_146], Original ATen: [aten.convolution, aten.silu]
        buf81 = extern_kernels.convolution(buf80, arg144_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=504, bias=None)
        assert_size_stride(buf81, (8, 504, 14, 14), (98784, 196, 14, 1))
        del arg144_1
        buf82 = buf81; del buf81  # reuse
        buf83 = empty_strided((8, 504, 1, 1), (504, 1, 4032, 4032), device='cuda', dtype=torch.float32)
        buf84 = reinterpret_tensor(buf83, (8, 504, 1, 1), (504, 1, 504, 504), 0); del buf83  # reuse
        # Source Nodes: [x_147, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_35.run(buf82, buf84, arg271_1, arg272_1, arg44_1, arg45_1, 4032, 196, grid=grid(4032), stream=stream0)
        del arg271_1
        del arg272_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
        buf85 = extern_kernels.convolution(buf84, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 42, 1, 1), (42, 1, 1, 1))
        del arg145_1
        del buf84
        buf86 = reinterpret_tensor(buf85, (8, 42, 1, 1), (42, 1, 42, 42), 0); del buf85  # reuse
        # Source Nodes: [getattr_l__mod___features___7___se_bn, x_se_16, x_se_17, x_se_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_36.run(buf86, arg146_1, arg337_1, arg338_1, arg147_1, arg148_1, 336, grid=grid(336), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        del arg337_1
        del arg338_1
        # Source Nodes: [getattr_l__mod___features___7___se_bn, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf87 = extern_kernels.convolution(buf86, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 504, 1, 1), (504, 1, 1, 1))
        del arg149_1
        del buf86
        buf88 = buf80; del buf80  # reuse
        # Source Nodes: [getattr_l__mod___features___7___se_bn, sigmoid_4, x_152, x_153, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_37.run(buf82, buf87, arg150_1, buf88, 4032, 196, grid=grid(4032, 196), stream=stream0)
        del arg150_1
        del buf82
        del buf87
        # Source Nodes: [getattr_l__mod___features___7___se_bn, sigmoid_4, x_152, x_153, x_154, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf89 = extern_kernels.convolution(buf88, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 95, 14, 14), (18620, 196, 14, 1))
        del arg151_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf90, arg273_1, arg274_1, arg46_1, arg47_1, 148960, grid=grid(148960), stream=stream0)
        del arg273_1
        del arg274_1
        del arg46_1
        del arg47_1
        buf91 = empty_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf90, buf77, buf91, 1568, 95, grid=grid(1568, 95), stream=stream0)
        del buf77
        del buf90
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 570, 14, 14), (111720, 196, 14, 1))
        del arg152_1
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf93, arg275_1, arg276_1, arg48_1, arg49_1, buf94, 4560, 196, grid=grid(4560, 196), stream=stream0)
        del arg275_1
        del arg276_1
        del arg48_1
        del arg49_1
        del buf93
        # Source Nodes: [x_166, x_167], Original ATen: [aten.convolution, aten.silu]
        buf95 = extern_kernels.convolution(buf94, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=570, bias=None)
        assert_size_stride(buf95, (8, 570, 14, 14), (111720, 196, 14, 1))
        del arg153_1
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided((8, 570, 1, 1), (570, 1, 4560, 4560), device='cuda', dtype=torch.float32)
        buf98 = reinterpret_tensor(buf97, (8, 570, 1, 1), (570, 1, 570, 570), 0); del buf97  # reuse
        # Source Nodes: [x_168, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_41.run(buf96, buf98, arg277_1, arg278_1, arg50_1, arg51_1, 4560, 196, grid=grid(4560), stream=stream0)
        del arg277_1
        del arg278_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
        buf99 = extern_kernels.convolution(buf98, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 47, 1, 1), (47, 1, 1, 1))
        del arg154_1
        del buf98
        buf100 = reinterpret_tensor(buf99, (8, 47, 1, 1), (47, 1, 47, 47), 0); del buf99  # reuse
        # Source Nodes: [getattr_l__mod___features___8___se_bn, x_se_20, x_se_21, x_se_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_42.run(buf100, arg155_1, arg340_1, arg341_1, arg156_1, arg157_1, 376, grid=grid(376), stream=stream0)
        del arg155_1
        del arg156_1
        del arg157_1
        del arg340_1
        del arg341_1
        # Source Nodes: [getattr_l__mod___features___8___se_bn, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf101 = extern_kernels.convolution(buf100, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 570, 1, 1), (570, 1, 1, 1))
        del arg158_1
        del buf100
        buf102 = buf94; del buf94  # reuse
        # Source Nodes: [getattr_l__mod___features___8___se_bn, sigmoid_5, x_173, x_174, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_43.run(buf96, buf101, arg159_1, buf102, 4560, 196, grid=grid(4560, 196), stream=stream0)
        del arg159_1
        del buf101
        del buf96
        # Source Nodes: [getattr_l__mod___features___8___se_bn, sigmoid_5, x_173, x_174, x_175, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf103 = extern_kernels.convolution(buf102, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 106, 14, 14), (20776, 196, 14, 1))
        del arg160_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf104, arg279_1, arg280_1, arg52_1, arg53_1, 166208, grid=grid(166208), stream=stream0)
        del arg279_1
        del arg280_1
        del arg52_1
        del arg53_1
        buf105 = empty_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf104, buf91, buf105, 1568, 106, grid=grid(1568, 106), stream=stream0)
        del buf104
        del buf91
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 636, 14, 14), (124656, 196, 14, 1))
        del arg161_1
        buf107 = buf106; del buf106  # reuse
        buf108 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_46.run(buf107, arg281_1, arg282_1, arg54_1, arg55_1, buf108, 5088, 196, grid=grid(5088, 196), stream=stream0)
        del arg281_1
        del arg282_1
        del arg54_1
        del arg55_1
        del buf107
        # Source Nodes: [x_187, x_188], Original ATen: [aten.convolution, aten.silu]
        buf109 = extern_kernels.convolution(buf108, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=636, bias=None)
        assert_size_stride(buf109, (8, 636, 14, 14), (124656, 196, 14, 1))
        del arg162_1
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided((8, 636, 1, 1), (636, 1, 5088, 5088), device='cuda', dtype=torch.float32)
        buf112 = reinterpret_tensor(buf111, (8, 636, 1, 1), (636, 1, 636, 636), 0); del buf111  # reuse
        # Source Nodes: [x_189, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_47.run(buf110, buf112, arg283_1, arg284_1, arg56_1, arg57_1, 5088, 196, grid=grid(5088), stream=stream0)
        del arg283_1
        del arg284_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
        buf113 = extern_kernels.convolution(buf112, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 53, 1, 1), (53, 1, 1, 1))
        del arg163_1
        del buf112
        buf114 = reinterpret_tensor(buf113, (8, 53, 1, 1), (53, 1, 53, 53), 0); del buf113  # reuse
        # Source Nodes: [getattr_l__mod___features___9___se_bn, x_se_24, x_se_25, x_se_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_48.run(buf114, arg164_1, arg343_1, arg344_1, arg165_1, arg166_1, 424, grid=grid(424), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        del arg343_1
        del arg344_1
        # Source Nodes: [getattr_l__mod___features___9___se_bn, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf115 = extern_kernels.convolution(buf114, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 636, 1, 1), (636, 1, 1, 1))
        del arg167_1
        del buf114
        buf116 = buf108; del buf108  # reuse
        # Source Nodes: [getattr_l__mod___features___9___se_bn, sigmoid_6, x_194, x_195, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_49.run(buf110, buf115, arg168_1, buf116, 5088, 196, grid=grid(5088, 196), stream=stream0)
        del arg168_1
        del buf110
        del buf115
        # Source Nodes: [getattr_l__mod___features___9___se_bn, sigmoid_6, x_194, x_195, x_196, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf117 = extern_kernels.convolution(buf116, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 117, 14, 14), (22932, 196, 14, 1))
        del arg169_1
        del buf116
        buf118 = buf117; del buf117  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf118, arg285_1, arg286_1, arg58_1, arg59_1, 183456, grid=grid(183456), stream=stream0)
        del arg285_1
        del arg286_1
        del arg58_1
        del arg59_1
        buf119 = empty_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf118, buf105, buf119, 1568, 117, grid=grid(1568, 117), stream=stream0)
        del buf105
        del buf118
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 702, 14, 14), (137592, 196, 14, 1))
        del arg170_1
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_52.run(buf121, arg287_1, arg288_1, arg60_1, arg61_1, buf122, 5616, 196, grid=grid(5616, 196), stream=stream0)
        del arg287_1
        del arg288_1
        del arg60_1
        del arg61_1
        del buf121
        # Source Nodes: [x_208, x_209], Original ATen: [aten.convolution, aten.silu]
        buf123 = extern_kernels.convolution(buf122, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=702, bias=None)
        assert_size_stride(buf123, (8, 702, 14, 14), (137592, 196, 14, 1))
        del arg171_1
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided((8, 702, 1, 1), (702, 1, 5616, 5616), device='cuda', dtype=torch.float32)
        buf126 = reinterpret_tensor(buf125, (8, 702, 1, 1), (702, 1, 702, 702), 0); del buf125  # reuse
        # Source Nodes: [x_210, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_53.run(buf124, buf126, arg289_1, arg290_1, arg62_1, arg63_1, 5616, 196, grid=grid(5616), stream=stream0)
        del arg289_1
        del arg290_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
        buf127 = extern_kernels.convolution(buf126, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 58, 1, 1), (58, 1, 1, 1))
        del arg172_1
        del buf126
        buf128 = reinterpret_tensor(buf127, (8, 58, 1, 1), (58, 1, 58, 58), 0); del buf127  # reuse
        # Source Nodes: [getattr_l__mod___features___10___se_bn, x_se_28, x_se_29, x_se_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54.run(buf128, arg173_1, arg346_1, arg347_1, arg174_1, arg175_1, 464, grid=grid(464), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        del arg346_1
        del arg347_1
        # Source Nodes: [getattr_l__mod___features___10___se_bn, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf129 = extern_kernels.convolution(buf128, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 702, 1, 1), (702, 1, 1, 1))
        del arg176_1
        del buf128
        buf130 = buf122; del buf122  # reuse
        # Source Nodes: [getattr_l__mod___features___10___se_bn, sigmoid_7, x_215, x_216, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55.run(buf124, buf129, arg177_1, buf130, 5616, 196, grid=grid(5616, 196), stream=stream0)
        del arg177_1
        del buf124
        del buf129
        # Source Nodes: [getattr_l__mod___features___10___se_bn, sigmoid_7, x_215, x_216, x_217, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf131 = extern_kernels.convolution(buf130, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg178_1
        del buf130
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf132, arg291_1, arg292_1, arg64_1, arg65_1, 200704, grid=grid(200704), stream=stream0)
        del arg291_1
        del arg292_1
        del arg64_1
        del arg65_1
        buf133 = empty_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(buf132, buf119, buf133, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf119
        del buf132
        # Source Nodes: [cat_15, x_224], Original ATen: [aten.cat, aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg179_1
        del buf133
        buf135 = buf134; del buf134  # reuse
        buf136 = reinterpret_tensor(buf0, (8, 768, 14, 14), (150528, 1, 10752, 768), 0); del buf0  # reuse
        # Source Nodes: [x_225, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_58.run(buf135, arg293_1, arg294_1, arg66_1, arg67_1, buf136, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg293_1
        del arg294_1
        del arg66_1
        del arg67_1
        del buf135
        # Source Nodes: [x_229, x_230], Original ATen: [aten.convolution, aten.silu]
        buf137 = extern_kernels.convolution(buf136, arg180_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf137, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg180_1
        del buf136
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf140 = reinterpret_tensor(buf139, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf139  # reuse
        # Source Nodes: [x_231, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_59.run(buf138, buf140, arg295_1, arg296_1, arg68_1, arg69_1, 6144, 49, grid=grid(6144), stream=stream0)
        del arg295_1
        del arg296_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
        buf141 = extern_kernels.convolution(buf140, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg181_1
        del buf140
        buf142 = reinterpret_tensor(buf141, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf141  # reuse
        # Source Nodes: [getattr_l__mod___features___11___se_bn, x_se_32, x_se_33, x_se_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_60.run(buf142, arg182_1, arg349_1, arg350_1, arg183_1, arg184_1, 512, grid=grid(512), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg349_1
        del arg350_1
        # Source Nodes: [getattr_l__mod___features___11___se_bn, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf143 = extern_kernels.convolution(buf142, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg185_1
        del buf142
        buf144 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___se_bn, sigmoid_8, x_236, x_237, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_61.run(buf138, buf143, arg186_1, buf144, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg186_1
        del buf138
        del buf143
        # Source Nodes: [getattr_l__mod___features___11___se_bn, sigmoid_8, x_236, x_237, x_238, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf145 = extern_kernels.convolution(buf144, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 140, 7, 7), (6860, 49, 7, 1))
        del arg187_1
        del buf144
        buf146 = empty_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_62.run(buf145, arg297_1, arg298_1, arg70_1, arg71_1, buf146, 1120, 49, grid=grid(1120, 49), stream=stream0)
        del arg297_1
        del arg298_1
        del arg70_1
        del arg71_1
        del buf145
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 840, 7, 7), (41160, 49, 7, 1))
        del arg188_1
        buf148 = buf147; del buf147  # reuse
        buf149 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_63.run(buf148, arg299_1, arg300_1, arg72_1, arg73_1, buf149, 6720, 49, grid=grid(6720, 49), stream=stream0)
        del arg299_1
        del arg300_1
        del arg72_1
        del arg73_1
        del buf148
        # Source Nodes: [x_249, x_250], Original ATen: [aten.convolution, aten.silu]
        buf150 = extern_kernels.convolution(buf149, arg189_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
        assert_size_stride(buf150, (8, 840, 7, 7), (41160, 49, 7, 1))
        del arg189_1
        buf151 = buf150; del buf150  # reuse
        buf152 = empty_strided((8, 840, 1, 1), (840, 1, 6720, 6720), device='cuda', dtype=torch.float32)
        buf153 = reinterpret_tensor(buf152, (8, 840, 1, 1), (840, 1, 840, 840), 0); del buf152  # reuse
        # Source Nodes: [x_251, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_64.run(buf151, buf153, arg301_1, arg302_1, arg74_1, arg75_1, 6720, 49, grid=grid(6720), stream=stream0)
        del arg301_1
        del arg302_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
        buf154 = extern_kernels.convolution(buf153, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 70, 1, 1), (70, 1, 1, 1))
        del arg190_1
        del buf153
        buf155 = reinterpret_tensor(buf154, (8, 70, 1, 1), (70, 1, 70, 70), 0); del buf154  # reuse
        # Source Nodes: [getattr_l__mod___features___12___se_bn, x_se_36, x_se_37, x_se_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65.run(buf155, arg191_1, arg352_1, arg353_1, arg192_1, arg193_1, 560, grid=grid(560), stream=stream0)
        del arg191_1
        del arg192_1
        del arg193_1
        del arg352_1
        del arg353_1
        # Source Nodes: [getattr_l__mod___features___12___se_bn, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf156 = extern_kernels.convolution(buf155, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 840, 1, 1), (840, 1, 1, 1))
        del arg194_1
        del buf155
        buf157 = buf149; del buf149  # reuse
        # Source Nodes: [getattr_l__mod___features___12___se_bn, sigmoid_9, x_256, x_257, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_66.run(buf151, buf156, arg195_1, buf157, 6720, 49, grid=grid(6720, 49), stream=stream0)
        del arg195_1
        del buf151
        del buf156
        # Source Nodes: [getattr_l__mod___features___12___se_bn, sigmoid_9, x_256, x_257, x_258, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf158 = extern_kernels.convolution(buf157, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 151, 7, 7), (7399, 49, 7, 1))
        del arg196_1
        del buf157
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_67.run(buf159, arg303_1, arg304_1, arg76_1, arg77_1, 59192, grid=grid(59192), stream=stream0)
        del arg303_1
        del arg304_1
        del arg76_1
        del arg77_1
        buf160 = empty_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf159, buf146, buf160, 392, 151, grid=grid(392, 151), stream=stream0)
        del buf146
        del buf159
        # Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 906, 7, 7), (44394, 49, 7, 1))
        del arg197_1
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_69.run(buf162, arg305_1, arg306_1, arg78_1, arg79_1, buf163, 7248, 49, grid=grid(7248, 49), stream=stream0)
        del arg305_1
        del arg306_1
        del arg78_1
        del arg79_1
        del buf162
        # Source Nodes: [x_270, x_271], Original ATen: [aten.convolution, aten.silu]
        buf164 = extern_kernels.convolution(buf163, arg198_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=906, bias=None)
        assert_size_stride(buf164, (8, 906, 7, 7), (44394, 49, 7, 1))
        del arg198_1
        buf165 = buf164; del buf164  # reuse
        buf166 = empty_strided((8, 906, 1, 1), (906, 1, 7248, 7248), device='cuda', dtype=torch.float32)
        buf167 = reinterpret_tensor(buf166, (8, 906, 1, 1), (906, 1, 906, 906), 0); del buf166  # reuse
        # Source Nodes: [x_272, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_70.run(buf165, buf167, arg307_1, arg308_1, arg80_1, arg81_1, 7248, 49, grid=grid(7248), stream=stream0)
        del arg307_1
        del arg308_1
        del arg80_1
        del arg81_1
        # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
        buf168 = extern_kernels.convolution(buf167, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 75, 1, 1), (75, 1, 1, 1))
        del arg199_1
        del buf167
        buf169 = reinterpret_tensor(buf168, (8, 75, 1, 1), (75, 1, 75, 75), 0); del buf168  # reuse
        # Source Nodes: [getattr_l__mod___features___13___se_bn, x_se_40, x_se_41, x_se_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_71.run(buf169, arg200_1, arg355_1, arg356_1, arg201_1, arg202_1, 600, grid=grid(600), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg355_1
        del arg356_1
        # Source Nodes: [getattr_l__mod___features___13___se_bn, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf170 = extern_kernels.convolution(buf169, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 906, 1, 1), (906, 1, 1, 1))
        del arg203_1
        del buf169
        buf171 = buf163; del buf163  # reuse
        # Source Nodes: [getattr_l__mod___features___13___se_bn, sigmoid_10, x_277, x_278, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_72.run(buf165, buf170, arg204_1, buf171, 7248, 49, grid=grid(7248, 49), stream=stream0)
        del arg204_1
        del buf165
        del buf170
        # Source Nodes: [getattr_l__mod___features___13___se_bn, sigmoid_10, x_277, x_278, x_279, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf172 = extern_kernels.convolution(buf171, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 162, 7, 7), (7938, 49, 7, 1))
        del arg205_1
        del buf171
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [x_280], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_73.run(buf173, arg309_1, arg310_1, arg82_1, arg83_1, 63504, grid=grid(63504), stream=stream0)
        del arg309_1
        del arg310_1
        del arg82_1
        del arg83_1
        buf174 = empty_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf173, buf160, buf174, 392, 162, grid=grid(392, 162), stream=stream0)
        del buf160
        del buf173
        # Source Nodes: [x_286], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 972, 7, 7), (47628, 49, 7, 1))
        del arg206_1
        buf176 = buf175; del buf175  # reuse
        buf177 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_75.run(buf176, arg311_1, arg312_1, arg84_1, arg85_1, buf177, 7776, 49, grid=grid(7776, 49), stream=stream0)
        del arg311_1
        del arg312_1
        del arg84_1
        del arg85_1
        del buf176
        # Source Nodes: [x_291, x_292], Original ATen: [aten.convolution, aten.silu]
        buf178 = extern_kernels.convolution(buf177, arg207_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=972, bias=None)
        assert_size_stride(buf178, (8, 972, 7, 7), (47628, 49, 7, 1))
        del arg207_1
        buf179 = buf178; del buf178  # reuse
        buf180 = empty_strided((8, 972, 1, 1), (972, 1, 7776, 7776), device='cuda', dtype=torch.float32)
        buf181 = reinterpret_tensor(buf180, (8, 972, 1, 1), (972, 1, 972, 972), 0); del buf180  # reuse
        # Source Nodes: [x_293, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_76.run(buf179, buf181, arg313_1, arg314_1, arg86_1, arg87_1, 7776, 49, grid=grid(7776), stream=stream0)
        del arg313_1
        del arg314_1
        del arg86_1
        del arg87_1
        # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
        buf182 = extern_kernels.convolution(buf181, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 81, 1, 1), (81, 1, 1, 1))
        del arg208_1
        del buf181
        buf183 = reinterpret_tensor(buf182, (8, 81, 1, 1), (81, 1, 81, 81), 0); del buf182  # reuse
        # Source Nodes: [getattr_l__mod___features___14___se_bn, x_se_44, x_se_45, x_se_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77.run(buf183, arg209_1, arg358_1, arg359_1, arg210_1, arg211_1, 648, grid=grid(648), stream=stream0)
        del arg209_1
        del arg210_1
        del arg211_1
        del arg358_1
        del arg359_1
        # Source Nodes: [getattr_l__mod___features___14___se_bn, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf184 = extern_kernels.convolution(buf183, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 972, 1, 1), (972, 1, 1, 1))
        del arg212_1
        del buf183
        buf185 = buf177; del buf177  # reuse
        # Source Nodes: [getattr_l__mod___features___14___se_bn, sigmoid_11, x_298, x_299, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78.run(buf179, buf184, arg213_1, buf185, 7776, 49, grid=grid(7776, 49), stream=stream0)
        del arg213_1
        del buf179
        del buf184
        # Source Nodes: [getattr_l__mod___features___14___se_bn, sigmoid_11, x_298, x_299, x_300, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf186 = extern_kernels.convolution(buf185, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 174, 7, 7), (8526, 49, 7, 1))
        del arg214_1
        del buf185
        buf187 = buf186; del buf186  # reuse
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_79.run(buf187, arg315_1, arg316_1, arg88_1, arg89_1, 68208, grid=grid(68208), stream=stream0)
        del arg315_1
        del arg316_1
        del arg88_1
        del arg89_1
        buf188 = empty_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_12], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf187, buf174, buf188, 392, 174, grid=grid(392, 174), stream=stream0)
        del buf174
        del buf187
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 1044, 7, 7), (51156, 49, 7, 1))
        del arg215_1
        buf190 = buf189; del buf189  # reuse
        buf191 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_81.run(buf190, arg317_1, arg318_1, arg90_1, arg91_1, buf191, 8352, 49, grid=grid(8352, 49), stream=stream0)
        del arg317_1
        del arg318_1
        del arg90_1
        del arg91_1
        del buf190
        # Source Nodes: [x_312, x_313], Original ATen: [aten.convolution, aten.silu]
        buf192 = extern_kernels.convolution(buf191, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1044, bias=None)
        assert_size_stride(buf192, (8, 1044, 7, 7), (51156, 49, 7, 1))
        del arg216_1
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided((8, 1044, 1, 1), (1044, 1, 8352, 8352), device='cuda', dtype=torch.float32)
        buf195 = reinterpret_tensor(buf194, (8, 1044, 1, 1), (1044, 1, 1044, 1044), 0); del buf194  # reuse
        # Source Nodes: [x_314, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_82.run(buf193, buf195, arg319_1, arg320_1, arg92_1, arg93_1, 8352, 49, grid=grid(8352), stream=stream0)
        del arg319_1
        del arg320_1
        del arg92_1
        del arg93_1
        # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
        buf196 = extern_kernels.convolution(buf195, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 87, 1, 1), (87, 1, 1, 1))
        del arg217_1
        del buf195
        buf197 = reinterpret_tensor(buf196, (8, 87, 1, 1), (87, 1, 87, 87), 0); del buf196  # reuse
        # Source Nodes: [getattr_l__mod___features___15___se_bn, x_se_48, x_se_49, x_se_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83.run(buf197, arg218_1, arg361_1, arg362_1, arg219_1, arg220_1, 696, grid=grid(696), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg361_1
        del arg362_1
        # Source Nodes: [getattr_l__mod___features___15___se_bn, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf198 = extern_kernels.convolution(buf197, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 1044, 1, 1), (1044, 1, 1, 1))
        del arg221_1
        del buf197
        buf199 = buf191; del buf191  # reuse
        # Source Nodes: [getattr_l__mod___features___15___se_bn, sigmoid_12, x_319, x_320, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84.run(buf193, buf198, arg222_1, buf199, 8352, 49, grid=grid(8352, 49), stream=stream0)
        del arg222_1
        del buf193
        del buf198
        # Source Nodes: [getattr_l__mod___features___15___se_bn, sigmoid_12, x_319, x_320, x_321, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf200 = extern_kernels.convolution(buf199, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 185, 7, 7), (9065, 49, 7, 1))
        del arg223_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_85.run(buf201, arg321_1, arg322_1, arg94_1, arg95_1, 72520, grid=grid(72520), stream=stream0)
        del arg321_1
        del arg322_1
        del arg94_1
        del arg95_1
        buf202 = empty_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_86.run(buf201, buf188, buf202, 392, 185, grid=grid(392, 185), stream=stream0)
        del buf188
        del buf201
        # Source Nodes: [cat_11, x_328], Original ATen: [aten.cat, aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 1280, 7, 7), (62720, 49, 7, 1))
        del arg224_1
        del buf202
        buf204 = buf203; del buf203  # reuse
        buf205 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf206 = reinterpret_tensor(buf205, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf205  # reuse
        # Source Nodes: [x_329, x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_87.run(buf204, buf206, arg323_1, arg324_1, arg96_1, arg97_1, 10240, 49, grid=grid(10240), stream=stream0)
        del arg323_1
        del arg324_1
        del arg96_1
        del arg97_1
        del buf204
        buf207 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg226_1, reinterpret_tensor(buf206, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg225_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf207)
        del arg225_1
        del arg226_1
        return (buf207, )


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
    arg10_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg328_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg331_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg334_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg337_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg340_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg343_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg346_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg349_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg352_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg355_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg358_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg361_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg364_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
