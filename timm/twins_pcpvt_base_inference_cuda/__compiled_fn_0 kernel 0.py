
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
# Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
# l__mod___patch_embeds_0_proj => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/sa/csagylybf5vxt53fsxptjwqluc77e7on55w7ea3mqr4letbkrk3a.py
# Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
# l__mod___patch_embeds_0_proj => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewu3bgiy6cuhg7bdxwjc65zguie3mwk2t5d25tu56pf3kclw2nr.py
# Source Nodes: [x_2], Original ATen: [aten.native_layer_norm]
# x_2 => clone, var_mean
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbcrumzrbddonxgtcpzsig3q53qliqyv22dsnrbbhtuy3nz6b6c.py
# Source Nodes: [x_2], Original ATen: [aten.native_layer_norm]
# x_2 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    x0 = xindex % 3136
    x2 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ywsqfb6a6euue3762hlah6p72bw4qptar4fl4po5z7gqaayrk4.py
# Source Nodes: [l__mod___blocks_0_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_0_norm1 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_per_fused_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7nheuxlzo4d6xb7qsf64t47iwny5m5wyeav4rqexee6jdnptbn.py
# Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
# l__mod___blocks_0_0_attn_sr => convolution_1
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (4096*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvw2wlkagikauvucxaavvhzpkj4gtr6jrngmqvfyeb2ysi63dxpw.py
# Source Nodes: [x_6], Original ATen: [aten.native_layer_norm]
# x_6 => add_4, add_5, clone_2, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (3136*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/cko2qmm4z6rh2qmt7fqnkj45b5ehlhmf5zpk5qt66bzxdmpui7bg.py
# Source Nodes: [l__mod___blocks_0_0_norm2, x_11], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_0_norm2 => add_7, add_8, mul_6, mul_7, rsqrt_3, sub_3, var_mean_3
# x_11 => add_6
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqlttlup2ipx4qlfcesg457dkpcetvoc3febg33rhw7qlsaihla.py
# Source Nodes: [x_13], Original ATen: [aten.gelu]
# x_13 => add_9, erf, mul_10, mul_8, mul_9
triton_poi_fused_gelu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3guzzpdv2filriydhm5dds7ht4ftubsqj6vlauqodoiyleun6qw.py
# Source Nodes: [x_11, x_19], Original ATen: [aten.add]
# x_11 => add_6
# x_19 => add_10
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rbsltvb3js7l6s4zos4ytxxdbrenorqxfhdlnncvhvvshtdksv.py
# Source Nodes: [l__mod___blocks_0_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_1_norm1 => add_12, add_13, clone_6, mul_11, mul_12, rsqrt_4, sub_4, var_mean_4
triton_per_fused_native_layer_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqukwoqj67agknvpki2ub5msyshti4sc7tnj4duoutmmbprlogpm.py
# Source Nodes: [l__mod___blocks_0_1_norm2, x_31], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_1_norm2 => add_17, add_18, clone_9, mul_15, mul_16, rsqrt_6, sub_6, var_mean_6
# x_31 => add_16
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6q3u6sqpabapszw3xfsn7tb6n6sbfuwxou2ttxdjung77hmum4i.py
# Source Nodes: [l__mod___blocks_0_2_norm1, x_39], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_2_norm1 => add_21, add_22, clone_12, mul_20, mul_21, rsqrt_7, sub_7, var_mean_7
# x_39 => add_20
triton_per_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqr5ifzqm5furnwchpucwdnsoy2fdkmmqmzc76767fcvevb45ka.py
# Source Nodes: [l__mod___blocks_0_2_norm2, x_39, x_47], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_2_norm2 => add_26, add_27, clone_15, mul_24, mul_25, rsqrt_9, sub_9, var_mean_9
# x_39 => add_20
# x_47 => add_25
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c7/cc7zomym2kvxeunsj5deftm5ldvgpgkk6fdvin3fwjjlq2ftsvip.py
# Source Nodes: [x_55], Original ATen: [aten.add]
# x_55 => add_29
triton_poi_fused_add_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crlme3d6n6a7m7734spjmoox5eduecdoazp2dlzuo4fe7sdeatsw.py
# Source Nodes: [l__mod___patch_embeds_1_proj], Original ATen: [aten.convolution]
# l__mod___patch_embeds_1_proj => convolution_5
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (256*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtwbpoaxglfpnzybvi3kedjjczprnlt22po5kufyaiz4x5dbqds.py
# Source Nodes: [x_59], Original ATen: [aten.native_layer_norm]
# x_59 => clone_18, var_mean_10
triton_red_fused_native_layer_norm_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvdiqwk7ooycxm4mwh2g6u5einq6ldzvjfsewyrmmipo2d2kyts.py
# Source Nodes: [x_59], Original ATen: [aten.native_layer_norm]
# x_59 => add_30, add_31, clone_18, mul_29, mul_30, rsqrt_10, sub_10, var_mean_10
triton_poi_fused_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 128
    x0 = xindex % 784
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctleag6tvg2stuseaxxqabqlqslkfj6lwjvbbsdwr7c5zkrmt34c.py
# Source Nodes: [l__mod___blocks_1_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_1_0_norm1 => add_32, add_33, mul_31, mul_32, rsqrt_11, sub_11, var_mean_11
triton_red_fused_native_layer_norm_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 128.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-06
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (x0 + (784*r2) + (100352*x1)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6xgiwwa6wewlcw4h6ntlvu43x43hxa47xa6j42abwywbnfnduh.py
# Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
# l__mod___blocks_1_0_attn_sr => convolution_6
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
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugxrqqdxqgbqfnvinnb2x2m3p4nedd6ge3ynsmtk7fgxilv7whs.py
# Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
# l__mod___blocks_1_0_attn_sr => convolution_6
triton_poi_fused_convolution_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (2048*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c432pphf6aa3ot7vakqdu3whaigmjxrojatzrzmwddz4662vxx2h.py
# Source Nodes: [x_63], Original ATen: [aten.native_layer_norm]
# x_63 => add_34, add_35, clone_20, mul_33, mul_34, rsqrt_12, sub_12, var_mean_12
triton_red_fused_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (49*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp20, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2i3c36wpw3kep2h74pmfxw6uwspyzbm3zsx5stbt5biazjmuxc.py
# Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
# l__mod___blocks_1_0_attn_q => addmm_15
triton_poi_fused_addmm_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (100352*(x0 // 784)) + (x0 % 784)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4uhk6lwgt6uoqzxujpz4yexrcokhzcpprcotqfhtmldmovfol6.py
# Source Nodes: [l__mod___blocks_1_0_norm2, x_68], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_0_norm2 => add_37, add_38, mul_35, mul_36, rsqrt_13, sub_13, var_mean_13
# x_68 => add_36
triton_red_fused_add_native_layer_norm_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqqz4xbnt4vfaczhe7oip3kmo2nnt5277lwu4ibf6vps7cetnid.py
# Source Nodes: [x_70], Original ATen: [aten.gelu]
# x_70 => add_39, erf_3, mul_37, mul_38, mul_39
triton_poi_fused_gelu_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgvqyibye476jv4xz4fjcf3obhrmxul5rcisgoqh2mwnwngavws.py
# Source Nodes: [x_68, x_76], Original ATen: [aten.add]
# x_68 => add_36
# x_76 => add_40
triton_poi_fused_add_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwuo6levthxnoflig5snrnzewlqkazdc3h62oxneixs3r6im2gqb.py
# Source Nodes: [l__mod___blocks_1_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_1_1_norm1 => add_42, add_43, clone_24, mul_40, mul_41, rsqrt_14, sub_14, var_mean_14
triton_red_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2trf62yk34zn6w33xbwgpwk6rnxhpttvdr67lsvqcd2mpeakusv.py
# Source Nodes: [l__mod___blocks_1_1_norm2, x_88], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_1_norm2 => add_47, add_48, clone_27, mul_44, mul_45, rsqrt_16, sub_16, var_mean_16
# x_88 => add_46
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5y2wi54xhbma6inulgb4odgeoknswuxmpvf6rg3aji5so5ffijw.py
# Source Nodes: [l__mod___blocks_1_2_norm1, x_96], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_2_norm1 => add_51, add_52, clone_30, mul_49, mul_50, rsqrt_17, sub_17, var_mean_17
# x_96 => add_50
triton_per_fused_add_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qqhdsqnep5dpmduz56e2y5kaewzviiel5igqnog4ohruabrzlx.py
# Source Nodes: [l__mod___blocks_1_2_norm2, x_104, x_96], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_2_norm2 => add_56, add_57, clone_33, mul_53, mul_54, rsqrt_19, sub_19, var_mean_19
# x_104 => add_55
# x_96 => add_50
triton_per_fused_add_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwm2lhua5ehil6oqxe67bt4puwa3eatzmaft7c7y5ezuwzk4erq5.py
# Source Nodes: [x_128], Original ATen: [aten.add]
# x_128 => add_68
triton_poi_fused_add_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5aup2sesa2trtenmlcqkwgcke7273yaogjizj4lqmkde5xeyunj.py
# Source Nodes: [l__mod___patch_embeds_2_proj], Original ATen: [aten.convolution]
# l__mod___patch_embeds_2_proj => convolution_11
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cekvzqrzvf66fgsq6tm6kpnqvja2thjjkzfyc5rzjmfib6ztovge.py
# Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
# x_132 => clone_42, var_mean_23
triton_red_fused_native_layer_norm_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 196) % 3
    x0 = xindex % 196
    x2 = (xindex // 588)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (196*r3) + (20972*x1) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp17, xmask)
    tl.store(out_ptr1 + (x4), tmp18, xmask)
    tl.store(out_ptr2 + (x4), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4hcn727hpft2453fvw3xj2t27cpyi33mwghtpxh2d32jfwwoy2.py
# Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
# x_132 => clone_42, var_mean_23
triton_per_fused_native_layer_norm_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27xoroxjverruxjv76qujhztftvhypexxesjbrtzuhy34idyfkm.py
# Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
# x_132 => add_69, add_70, clone_42, mul_67, mul_68, rsqrt_23, sub_23, var_mean_23
triton_poi_fused_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 320
    x0 = xindex % 196
    x2 = (xindex // 62720)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 320.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnvyf6uroywnxsod4g7lw4qtqajz4j3aitkpxwzq2k7jc4echubk.py
# Source Nodes: [l__mod___blocks_2_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_2_0_norm1 => var_mean_24
triton_red_fused_native_layer_norm_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (196*x0) + (588*x2)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (196*x0) + (588*x2)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (196*x0) + (588*x2)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7okqu32royfktqdi67vcsoduhqdwpyzdfx3vuahnhfoto6ihud.py
# Source Nodes: [l__mod___blocks_2_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_2_0_norm1 => add_71, add_72, mul_69, mul_70, rsqrt_24, sub_24, var_mean_24
triton_poi_fused_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 320.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (196*x2) + (62720*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxkz3mylvtjrjdy6dzwfw76cdjnuhqkw7fhtjl6xdk4y74je62z.py
# Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
# l__mod___blocks_2_0_attn_sr => convolution_12
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (62720*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7op4nnhpo7gq7e2rosik4v36gcwci523cnsi2cmjldtv4ye6ys.py
# Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
# l__mod___blocks_2_0_attn_sr => convolution_12
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd5wbupgcf4o567h3yff4w4j5nnkyseayr6e22biuedwgu3w25t.py
# Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
# x_136 => clone_44, var_mean_25
triton_red_fused_native_layer_norm_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1176
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49) % 3
    x0 = xindex % 49
    x2 = (xindex // 147)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (49*r3) + (5243*x1) + (15680*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp17, xmask)
    tl.store(out_ptr1 + (x4), tmp18, xmask)
    tl.store(out_ptr2 + (x4), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdsyfefa46qzrnddwajjpxkhth3kthzxqcfczbbbc7lpdznt4xv.py
# Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
# x_136 => clone_44, var_mean_25
triton_per_fused_native_layer_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (147*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqn7fdrb6iaqt7f57lqmiasiii3gsb7cb7bo2canny3ngdk6tk2g.py
# Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
# x_136 => add_73, add_74, clone_44, mul_71, mul_72, rsqrt_25, sub_25, var_mean_25
triton_poi_fused_native_layer_norm_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 320.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cex3wxuhzhk4b2o5do67fxuaoulk6uhnce2iuvfl2yatcyop734g.py
# Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
# l__mod___blocks_2_0_attn_q => addmm_35
triton_poi_fused_addmm_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (62720*(x0 // 196)) + (x0 % 196)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcdivroq4ur2hmgbw32bhksiq4xer7yraykspkgxb72yqyagscp.py
# Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_0_norm2 => var_mean_26
# x_141 => add_75
triton_red_fused_add_native_layer_norm_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x4 = (xindex // 3)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x0) + (320*x4)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r3 + (107*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp19, xmask)
    tl.store(out_ptr1 + (x5), tmp20, xmask)
    tl.store(out_ptr2 + (x5), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6es7ba25zn3k65thorfcsjrsubc3qot6gjqavlryjchysdiui7j.py
# Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_0_norm2 => var_mean_26
# x_141 => add_75
triton_per_fused_add_native_layer_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshrnkyhuongc6nfcxzqyyzv45ayzh2zrrkawlicsi3p5el2a6tq.py
# Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_0_norm2 => add_76, add_77, mul_73, mul_74, rsqrt_26, sub_26, var_mean_26
# x_141 => add_75
triton_poi_fused_add_native_layer_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 320.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4dgylrcpf2ev2cmbficgm5ssnyipst7tlgxwdp4jcudoopqqz2.py
# Source Nodes: [x_143], Original ATen: [aten.gelu]
# x_143 => add_78, erf_7, mul_75, mul_76, mul_77
triton_poi_fused_gelu_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62pkngzeg7lxnwhcrfg3pp5z4pnh2pk5odtsqcnsy24fd6kt56u.py
# Source Nodes: [x_141, x_149], Original ATen: [aten.add]
# x_141 => add_75
# x_149 => add_79
triton_poi_fused_add_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (320*y3)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4knb6el5mibzuyuujcpditdykha44axahctmy2pclhmk5tafgfg.py
# Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_2_1_norm1 => clone_48, var_mean_27
triton_red_fused_native_layer_norm_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x4 = (xindex // 3)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r3 + (107*x0) + (320*x4)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp19, xmask)
    tl.store(out_ptr1 + (x5), tmp20, xmask)
    tl.store(out_ptr2 + (x5), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/cticbfvxbgk5wrbudtstoo7w3ra5qlvg4obbblri7laak6trvg6b.py
# Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_2_1_norm1 => add_81, add_82, clone_48, mul_78, mul_79, rsqrt_27, sub_27, var_mean_27
triton_poi_fused_native_layer_norm_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 320.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlczhk3jiww3xjvxbmayqgha5hwief4jsaeoma3gpcidmtuehv6.py
# Source Nodes: [l__mod___blocks_2_1_norm2, x_161], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_1_norm2 => add_86, add_87, clone_51, mul_82, mul_83, rsqrt_29, sub_29, var_mean_29
# x_161 => add_85
triton_per_fused_add_native_layer_norm_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (320*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbvaq3nudd2p5vridb6nsgd2ci6idwlcrfbn5jhap4m4ucxhifk.py
# Source Nodes: [l__mod___blocks_2_2_norm1, x_169], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_2_norm1 => add_90, add_91, clone_54, mul_87, mul_88, rsqrt_30, sub_30, var_mean_30
# x_169 => add_89
triton_per_fused_add_native_layer_norm_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 320, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 320.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctctdwlu6eouo3yhnysseyj3tje75vy7ydg4ezz6v56qn3jiub7d.py
# Source Nodes: [l__mod___blocks_2_2_norm2, x_169, x_177], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_2_2_norm2 => add_95, add_96, clone_57, mul_91, mul_92, rsqrt_32, sub_32, var_mean_32
# x_169 => add_89
# x_177 => add_94
triton_per_fused_add_native_layer_norm_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6q7qf5t64r23uhl3pjgbpwgjbvzyusxmcpq7ualfv4xg6ov2wmb.py
# Source Nodes: [x_425], Original ATen: [aten.add]
# x_425 => add_233
triton_poi_fused_add_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpeo5p2mflz6rx3iofz6iedmhxktajebqwgshbwx37qlkdftsnkk.py
# Source Nodes: [l__mod___patch_embeds_3_proj], Original ATen: [aten.convolution]
# l__mod___patch_embeds_3_proj => convolution_31
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 163840
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4shallmkvud5smjc5udg6igfdpgo25x4titw5t6bfle3ac4z7l.py
# Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
# x_429 => clone_150, var_mean_78
triton_red_fused_native_layer_norm_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r3) + (6272*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7h7wfc3zgvyk4apg2u5fr4vnh3r4j2btixktrrkmfpbrmdhysb.py
# Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
# x_429 => clone_150, var_mean_78
triton_per_fused_native_layer_norm_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnoy5nqkmsknrhci3qlkehlq3n6g7ibpbil2opx7gnkelsiplcb.py
# Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
# x_429 => add_234, add_235, clone_150, mul_231, mul_232, rsqrt_78, sub_78, var_mean_78
triton_poi_fused_native_layer_norm_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 512
    x0 = xindex % 49
    x2 = (xindex // 25088)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (49*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + (49*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chadajwua3kjlnp7q6z3tf74ksbmyqeheonq4ek5gnznfclrmca4.py
# Source Nodes: [l__mod___blocks_3_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_3_0_norm1 => var_mean_79
triton_red_fused_native_layer_norm_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (49*x0) + (196*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (49*x0) + (196*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (49*x0) + (196*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvwd3bh7jynf6xzywaoammr5aozvm23ief4c6eypjme3uu2jhte.py
# Source Nodes: [l__mod___blocks_3_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_3_0_norm1 => add_236, add_237, mul_233, mul_234, rsqrt_79, sub_79, var_mean_79
triton_poi_fused_native_layer_norm_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wt/cwtu5d3zdnnjq6xsrp7qztxvjargeetvjf6hblbslhean7m7olzh.py
# Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_0_norm2 => var_mean_80
# x_435 => add_238
triton_red_fused_add_native_layer_norm_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6uaram3edemanow2pgln4jzmwccs47hg3bk5raiflllsrrkchs.py
# Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_0_norm2 => var_mean_80
# x_435 => add_238
triton_per_fused_add_native_layer_norm_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33fjrllnh6dhgys4t6vewx357ts3cmmahxmehv6jgdp2torsyrv.py
# Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_0_norm2 => add_239, add_240, mul_235, mul_236, rsqrt_80, sub_80, var_mean_80
# x_435 => add_238
triton_poi_fused_add_native_layer_norm_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nm37xpjbxpl4cjsxg7zriqeml2krcjgan4444neydwv4a23rgg.py
# Source Nodes: [x_437], Original ATen: [aten.gelu]
# x_437 => add_241, erf_25, mul_237, mul_238, mul_239
triton_poi_fused_gelu_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2ngteazxsgyvvdldlqnlzy7d5xb42pwtkm4eklfi2bltr2wyyp.py
# Source Nodes: [x_435, x_443], Original ATen: [aten.add]
# x_435 => add_238
# x_443 => add_242
triton_poi_fused_add_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3tsaax6heix3cntdko5psoag4sh7pzp2ag5gbrm6eazx2jjz4q.py
# Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_3_1_norm1 => clone_155, var_mean_81
triton_red_fused_native_layer_norm_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    x5 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
    tl.store(out_ptr1 + (x5), tmp7, xmask)
    tl.store(out_ptr2 + (x5), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpfnuwlfg5h5vhrhosr7ztoidsuz3medppjxazr46gvf7imddux.py
# Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_3_1_norm1 => add_244, add_245, clone_155, mul_240, mul_241, rsqrt_81, sub_81, var_mean_81
triton_poi_fused_native_layer_norm_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljlmurv22nbr3jcfq3fm3zmfytyzmzmyvo4gdl6yfil4xn3qfq7.py
# Source Nodes: [l__mod___blocks_3_1_norm2, x_452], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_1_norm2 => add_247, add_248, clone_157, mul_242, mul_243, rsqrt_82, sub_82, var_mean_82
# x_452 => add_246
triton_per_fused_add_native_layer_norm_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (25088*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eo/ceowbltx2utwsppeag5odvgr3g423laorrjkzt5yyqpmws4zhehu.py
# Source Nodes: [l__mod___blocks_3_2_norm1, x_460], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_2_norm1 => add_251, add_252, clone_160, mul_247, mul_248, rsqrt_83, sub_83, var_mean_83
# x_460 => add_250
triton_per_fused_add_native_layer_norm_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/au/cauwkzyw7n3zasilgd2i4o7gqo2qk5ayc7ck62c6arxdrj4egzmp.py
# Source Nodes: [l__mod___blocks_3_2_norm2, x_460, x_465], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_3_2_norm2 => add_254, add_255, clone_162, mul_249, mul_250, rsqrt_84, sub_84, var_mean_84
# x_460 => add_250
# x_465 => add_253
triton_per_fused_add_native_layer_norm_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_69', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhad67kzww52guy5vayumfv4jzm377lpficcexretwbzn3z4cdc.py
# Source Nodes: [x_473, x_475], Original ATen: [aten.add, aten.native_layer_norm]
# x_473 => add_257
# x_475 => clone_165, var_mean_85
triton_per_fused_add_native_layer_norm_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clijdpz2cyr64dmp5wfsnxwsnawx3s6l4qwk4p473t44ygvr5h5q.py
# Source Nodes: [x_473, x_475, x_476], Original ATen: [aten.add, aten.mean, aten.native_layer_norm]
# x_473 => add_257
# x_475 => add_258, add_259, clone_165, mul_254, mul_255, rsqrt_85, sub_85, var_mean_85
# x_476 => mean
triton_per_fused_add_mean_native_layer_norm_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_71', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (25088*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (25088*x1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64), (64, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (128, 64), (64, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (64, 64), (64, 1))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (512, 64), (64, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (64, 512), (512, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, 64), (64, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (128, 64), (64, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (64, 64), (64, 1))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (512, 64), (64, 1))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (64, 512), (512, 1))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, 64), (64, 1))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (128, 64), (64, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (64, 64), (64, 1))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (512, 64), (64, 1))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (64, 512), (512, 1))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128), (128, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (256, 128), (128, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (128, 128), (128, 1))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (1024, 128), (128, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (128, 1024), (1024, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, 128), (128, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (256, 128), (128, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (128, 128), (128, 1))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (1024, 128), (128, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (128, 1024), (1024, 1))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, 128), (128, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (256, 128), (128, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (128, 128), (128, 1))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (1024, 128), (128, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (128, 1024), (1024, 1))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, 128), (128, 1))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (256, 128), (128, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (128, 128), (128, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (1024, 128), (128, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (128, 1024), (1024, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg139_1, (320, ), (1, ))
    assert_size_stride(arg140_1, (320, ), (1, ))
    assert_size_stride(arg141_1, (320, ), (1, ))
    assert_size_stride(arg142_1, (320, ), (1, ))
    assert_size_stride(arg143_1, (320, ), (1, ))
    assert_size_stride(arg144_1, (320, 320), (320, 1))
    assert_size_stride(arg145_1, (320, ), (1, ))
    assert_size_stride(arg146_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg147_1, (320, ), (1, ))
    assert_size_stride(arg148_1, (320, ), (1, ))
    assert_size_stride(arg149_1, (320, ), (1, ))
    assert_size_stride(arg150_1, (640, 320), (320, 1))
    assert_size_stride(arg151_1, (640, ), (1, ))
    assert_size_stride(arg152_1, (320, 320), (320, 1))
    assert_size_stride(arg153_1, (320, ), (1, ))
    assert_size_stride(arg154_1, (320, ), (1, ))
    assert_size_stride(arg155_1, (320, ), (1, ))
    assert_size_stride(arg156_1, (1280, 320), (320, 1))
    assert_size_stride(arg157_1, (1280, ), (1, ))
    assert_size_stride(arg158_1, (320, 1280), (1280, 1))
    assert_size_stride(arg159_1, (320, ), (1, ))
    assert_size_stride(arg160_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg161_1, (320, ), (1, ))
    assert_size_stride(arg162_1, (320, ), (1, ))
    assert_size_stride(arg163_1, (320, ), (1, ))
    assert_size_stride(arg164_1, (320, 320), (320, 1))
    assert_size_stride(arg165_1, (320, ), (1, ))
    assert_size_stride(arg166_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg167_1, (320, ), (1, ))
    assert_size_stride(arg168_1, (320, ), (1, ))
    assert_size_stride(arg169_1, (320, ), (1, ))
    assert_size_stride(arg170_1, (640, 320), (320, 1))
    assert_size_stride(arg171_1, (640, ), (1, ))
    assert_size_stride(arg172_1, (320, 320), (320, 1))
    assert_size_stride(arg173_1, (320, ), (1, ))
    assert_size_stride(arg174_1, (320, ), (1, ))
    assert_size_stride(arg175_1, (320, ), (1, ))
    assert_size_stride(arg176_1, (1280, 320), (320, 1))
    assert_size_stride(arg177_1, (1280, ), (1, ))
    assert_size_stride(arg178_1, (320, 1280), (1280, 1))
    assert_size_stride(arg179_1, (320, ), (1, ))
    assert_size_stride(arg180_1, (320, ), (1, ))
    assert_size_stride(arg181_1, (320, ), (1, ))
    assert_size_stride(arg182_1, (320, 320), (320, 1))
    assert_size_stride(arg183_1, (320, ), (1, ))
    assert_size_stride(arg184_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg185_1, (320, ), (1, ))
    assert_size_stride(arg186_1, (320, ), (1, ))
    assert_size_stride(arg187_1, (320, ), (1, ))
    assert_size_stride(arg188_1, (640, 320), (320, 1))
    assert_size_stride(arg189_1, (640, ), (1, ))
    assert_size_stride(arg190_1, (320, 320), (320, 1))
    assert_size_stride(arg191_1, (320, ), (1, ))
    assert_size_stride(arg192_1, (320, ), (1, ))
    assert_size_stride(arg193_1, (320, ), (1, ))
    assert_size_stride(arg194_1, (1280, 320), (320, 1))
    assert_size_stride(arg195_1, (1280, ), (1, ))
    assert_size_stride(arg196_1, (320, 1280), (1280, 1))
    assert_size_stride(arg197_1, (320, ), (1, ))
    assert_size_stride(arg198_1, (320, ), (1, ))
    assert_size_stride(arg199_1, (320, ), (1, ))
    assert_size_stride(arg200_1, (320, 320), (320, 1))
    assert_size_stride(arg201_1, (320, ), (1, ))
    assert_size_stride(arg202_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg203_1, (320, ), (1, ))
    assert_size_stride(arg204_1, (320, ), (1, ))
    assert_size_stride(arg205_1, (320, ), (1, ))
    assert_size_stride(arg206_1, (640, 320), (320, 1))
    assert_size_stride(arg207_1, (640, ), (1, ))
    assert_size_stride(arg208_1, (320, 320), (320, 1))
    assert_size_stride(arg209_1, (320, ), (1, ))
    assert_size_stride(arg210_1, (320, ), (1, ))
    assert_size_stride(arg211_1, (320, ), (1, ))
    assert_size_stride(arg212_1, (1280, 320), (320, 1))
    assert_size_stride(arg213_1, (1280, ), (1, ))
    assert_size_stride(arg214_1, (320, 1280), (1280, 1))
    assert_size_stride(arg215_1, (320, ), (1, ))
    assert_size_stride(arg216_1, (320, ), (1, ))
    assert_size_stride(arg217_1, (320, ), (1, ))
    assert_size_stride(arg218_1, (320, 320), (320, 1))
    assert_size_stride(arg219_1, (320, ), (1, ))
    assert_size_stride(arg220_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg221_1, (320, ), (1, ))
    assert_size_stride(arg222_1, (320, ), (1, ))
    assert_size_stride(arg223_1, (320, ), (1, ))
    assert_size_stride(arg224_1, (640, 320), (320, 1))
    assert_size_stride(arg225_1, (640, ), (1, ))
    assert_size_stride(arg226_1, (320, 320), (320, 1))
    assert_size_stride(arg227_1, (320, ), (1, ))
    assert_size_stride(arg228_1, (320, ), (1, ))
    assert_size_stride(arg229_1, (320, ), (1, ))
    assert_size_stride(arg230_1, (1280, 320), (320, 1))
    assert_size_stride(arg231_1, (1280, ), (1, ))
    assert_size_stride(arg232_1, (320, 1280), (1280, 1))
    assert_size_stride(arg233_1, (320, ), (1, ))
    assert_size_stride(arg234_1, (320, ), (1, ))
    assert_size_stride(arg235_1, (320, ), (1, ))
    assert_size_stride(arg236_1, (320, 320), (320, 1))
    assert_size_stride(arg237_1, (320, ), (1, ))
    assert_size_stride(arg238_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg239_1, (320, ), (1, ))
    assert_size_stride(arg240_1, (320, ), (1, ))
    assert_size_stride(arg241_1, (320, ), (1, ))
    assert_size_stride(arg242_1, (640, 320), (320, 1))
    assert_size_stride(arg243_1, (640, ), (1, ))
    assert_size_stride(arg244_1, (320, 320), (320, 1))
    assert_size_stride(arg245_1, (320, ), (1, ))
    assert_size_stride(arg246_1, (320, ), (1, ))
    assert_size_stride(arg247_1, (320, ), (1, ))
    assert_size_stride(arg248_1, (1280, 320), (320, 1))
    assert_size_stride(arg249_1, (1280, ), (1, ))
    assert_size_stride(arg250_1, (320, 1280), (1280, 1))
    assert_size_stride(arg251_1, (320, ), (1, ))
    assert_size_stride(arg252_1, (320, ), (1, ))
    assert_size_stride(arg253_1, (320, ), (1, ))
    assert_size_stride(arg254_1, (320, 320), (320, 1))
    assert_size_stride(arg255_1, (320, ), (1, ))
    assert_size_stride(arg256_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg257_1, (320, ), (1, ))
    assert_size_stride(arg258_1, (320, ), (1, ))
    assert_size_stride(arg259_1, (320, ), (1, ))
    assert_size_stride(arg260_1, (640, 320), (320, 1))
    assert_size_stride(arg261_1, (640, ), (1, ))
    assert_size_stride(arg262_1, (320, 320), (320, 1))
    assert_size_stride(arg263_1, (320, ), (1, ))
    assert_size_stride(arg264_1, (320, ), (1, ))
    assert_size_stride(arg265_1, (320, ), (1, ))
    assert_size_stride(arg266_1, (1280, 320), (320, 1))
    assert_size_stride(arg267_1, (1280, ), (1, ))
    assert_size_stride(arg268_1, (320, 1280), (1280, 1))
    assert_size_stride(arg269_1, (320, ), (1, ))
    assert_size_stride(arg270_1, (320, ), (1, ))
    assert_size_stride(arg271_1, (320, ), (1, ))
    assert_size_stride(arg272_1, (320, 320), (320, 1))
    assert_size_stride(arg273_1, (320, ), (1, ))
    assert_size_stride(arg274_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg275_1, (320, ), (1, ))
    assert_size_stride(arg276_1, (320, ), (1, ))
    assert_size_stride(arg277_1, (320, ), (1, ))
    assert_size_stride(arg278_1, (640, 320), (320, 1))
    assert_size_stride(arg279_1, (640, ), (1, ))
    assert_size_stride(arg280_1, (320, 320), (320, 1))
    assert_size_stride(arg281_1, (320, ), (1, ))
    assert_size_stride(arg282_1, (320, ), (1, ))
    assert_size_stride(arg283_1, (320, ), (1, ))
    assert_size_stride(arg284_1, (1280, 320), (320, 1))
    assert_size_stride(arg285_1, (1280, ), (1, ))
    assert_size_stride(arg286_1, (320, 1280), (1280, 1))
    assert_size_stride(arg287_1, (320, ), (1, ))
    assert_size_stride(arg288_1, (320, ), (1, ))
    assert_size_stride(arg289_1, (320, ), (1, ))
    assert_size_stride(arg290_1, (320, 320), (320, 1))
    assert_size_stride(arg291_1, (320, ), (1, ))
    assert_size_stride(arg292_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg293_1, (320, ), (1, ))
    assert_size_stride(arg294_1, (320, ), (1, ))
    assert_size_stride(arg295_1, (320, ), (1, ))
    assert_size_stride(arg296_1, (640, 320), (320, 1))
    assert_size_stride(arg297_1, (640, ), (1, ))
    assert_size_stride(arg298_1, (320, 320), (320, 1))
    assert_size_stride(arg299_1, (320, ), (1, ))
    assert_size_stride(arg300_1, (320, ), (1, ))
    assert_size_stride(arg301_1, (320, ), (1, ))
    assert_size_stride(arg302_1, (1280, 320), (320, 1))
    assert_size_stride(arg303_1, (1280, ), (1, ))
    assert_size_stride(arg304_1, (320, 1280), (1280, 1))
    assert_size_stride(arg305_1, (320, ), (1, ))
    assert_size_stride(arg306_1, (320, ), (1, ))
    assert_size_stride(arg307_1, (320, ), (1, ))
    assert_size_stride(arg308_1, (320, 320), (320, 1))
    assert_size_stride(arg309_1, (320, ), (1, ))
    assert_size_stride(arg310_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg311_1, (320, ), (1, ))
    assert_size_stride(arg312_1, (320, ), (1, ))
    assert_size_stride(arg313_1, (320, ), (1, ))
    assert_size_stride(arg314_1, (640, 320), (320, 1))
    assert_size_stride(arg315_1, (640, ), (1, ))
    assert_size_stride(arg316_1, (320, 320), (320, 1))
    assert_size_stride(arg317_1, (320, ), (1, ))
    assert_size_stride(arg318_1, (320, ), (1, ))
    assert_size_stride(arg319_1, (320, ), (1, ))
    assert_size_stride(arg320_1, (1280, 320), (320, 1))
    assert_size_stride(arg321_1, (1280, ), (1, ))
    assert_size_stride(arg322_1, (320, 1280), (1280, 1))
    assert_size_stride(arg323_1, (320, ), (1, ))
    assert_size_stride(arg324_1, (320, ), (1, ))
    assert_size_stride(arg325_1, (320, ), (1, ))
    assert_size_stride(arg326_1, (320, 320), (320, 1))
    assert_size_stride(arg327_1, (320, ), (1, ))
    assert_size_stride(arg328_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg329_1, (320, ), (1, ))
    assert_size_stride(arg330_1, (320, ), (1, ))
    assert_size_stride(arg331_1, (320, ), (1, ))
    assert_size_stride(arg332_1, (640, 320), (320, 1))
    assert_size_stride(arg333_1, (640, ), (1, ))
    assert_size_stride(arg334_1, (320, 320), (320, 1))
    assert_size_stride(arg335_1, (320, ), (1, ))
    assert_size_stride(arg336_1, (320, ), (1, ))
    assert_size_stride(arg337_1, (320, ), (1, ))
    assert_size_stride(arg338_1, (1280, 320), (320, 1))
    assert_size_stride(arg339_1, (1280, ), (1, ))
    assert_size_stride(arg340_1, (320, 1280), (1280, 1))
    assert_size_stride(arg341_1, (320, ), (1, ))
    assert_size_stride(arg342_1, (320, ), (1, ))
    assert_size_stride(arg343_1, (320, ), (1, ))
    assert_size_stride(arg344_1, (320, 320), (320, 1))
    assert_size_stride(arg345_1, (320, ), (1, ))
    assert_size_stride(arg346_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg347_1, (320, ), (1, ))
    assert_size_stride(arg348_1, (320, ), (1, ))
    assert_size_stride(arg349_1, (320, ), (1, ))
    assert_size_stride(arg350_1, (640, 320), (320, 1))
    assert_size_stride(arg351_1, (640, ), (1, ))
    assert_size_stride(arg352_1, (320, 320), (320, 1))
    assert_size_stride(arg353_1, (320, ), (1, ))
    assert_size_stride(arg354_1, (320, ), (1, ))
    assert_size_stride(arg355_1, (320, ), (1, ))
    assert_size_stride(arg356_1, (1280, 320), (320, 1))
    assert_size_stride(arg357_1, (1280, ), (1, ))
    assert_size_stride(arg358_1, (320, 1280), (1280, 1))
    assert_size_stride(arg359_1, (320, ), (1, ))
    assert_size_stride(arg360_1, (320, ), (1, ))
    assert_size_stride(arg361_1, (320, ), (1, ))
    assert_size_stride(arg362_1, (320, 320), (320, 1))
    assert_size_stride(arg363_1, (320, ), (1, ))
    assert_size_stride(arg364_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg365_1, (320, ), (1, ))
    assert_size_stride(arg366_1, (320, ), (1, ))
    assert_size_stride(arg367_1, (320, ), (1, ))
    assert_size_stride(arg368_1, (640, 320), (320, 1))
    assert_size_stride(arg369_1, (640, ), (1, ))
    assert_size_stride(arg370_1, (320, 320), (320, 1))
    assert_size_stride(arg371_1, (320, ), (1, ))
    assert_size_stride(arg372_1, (320, ), (1, ))
    assert_size_stride(arg373_1, (320, ), (1, ))
    assert_size_stride(arg374_1, (1280, 320), (320, 1))
    assert_size_stride(arg375_1, (1280, ), (1, ))
    assert_size_stride(arg376_1, (320, 1280), (1280, 1))
    assert_size_stride(arg377_1, (320, ), (1, ))
    assert_size_stride(arg378_1, (320, ), (1, ))
    assert_size_stride(arg379_1, (320, ), (1, ))
    assert_size_stride(arg380_1, (320, 320), (320, 1))
    assert_size_stride(arg381_1, (320, ), (1, ))
    assert_size_stride(arg382_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg383_1, (320, ), (1, ))
    assert_size_stride(arg384_1, (320, ), (1, ))
    assert_size_stride(arg385_1, (320, ), (1, ))
    assert_size_stride(arg386_1, (640, 320), (320, 1))
    assert_size_stride(arg387_1, (640, ), (1, ))
    assert_size_stride(arg388_1, (320, 320), (320, 1))
    assert_size_stride(arg389_1, (320, ), (1, ))
    assert_size_stride(arg390_1, (320, ), (1, ))
    assert_size_stride(arg391_1, (320, ), (1, ))
    assert_size_stride(arg392_1, (1280, 320), (320, 1))
    assert_size_stride(arg393_1, (1280, ), (1, ))
    assert_size_stride(arg394_1, (320, 1280), (1280, 1))
    assert_size_stride(arg395_1, (320, ), (1, ))
    assert_size_stride(arg396_1, (320, ), (1, ))
    assert_size_stride(arg397_1, (320, ), (1, ))
    assert_size_stride(arg398_1, (320, 320), (320, 1))
    assert_size_stride(arg399_1, (320, ), (1, ))
    assert_size_stride(arg400_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg401_1, (320, ), (1, ))
    assert_size_stride(arg402_1, (320, ), (1, ))
    assert_size_stride(arg403_1, (320, ), (1, ))
    assert_size_stride(arg404_1, (640, 320), (320, 1))
    assert_size_stride(arg405_1, (640, ), (1, ))
    assert_size_stride(arg406_1, (320, 320), (320, 1))
    assert_size_stride(arg407_1, (320, ), (1, ))
    assert_size_stride(arg408_1, (320, ), (1, ))
    assert_size_stride(arg409_1, (320, ), (1, ))
    assert_size_stride(arg410_1, (1280, 320), (320, 1))
    assert_size_stride(arg411_1, (1280, ), (1, ))
    assert_size_stride(arg412_1, (320, 1280), (1280, 1))
    assert_size_stride(arg413_1, (320, ), (1, ))
    assert_size_stride(arg414_1, (320, ), (1, ))
    assert_size_stride(arg415_1, (320, ), (1, ))
    assert_size_stride(arg416_1, (320, 320), (320, 1))
    assert_size_stride(arg417_1, (320, ), (1, ))
    assert_size_stride(arg418_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg419_1, (320, ), (1, ))
    assert_size_stride(arg420_1, (320, ), (1, ))
    assert_size_stride(arg421_1, (320, ), (1, ))
    assert_size_stride(arg422_1, (640, 320), (320, 1))
    assert_size_stride(arg423_1, (640, ), (1, ))
    assert_size_stride(arg424_1, (320, 320), (320, 1))
    assert_size_stride(arg425_1, (320, ), (1, ))
    assert_size_stride(arg426_1, (320, ), (1, ))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (1280, 320), (320, 1))
    assert_size_stride(arg429_1, (1280, ), (1, ))
    assert_size_stride(arg430_1, (320, 1280), (1280, 1))
    assert_size_stride(arg431_1, (320, ), (1, ))
    assert_size_stride(arg432_1, (320, ), (1, ))
    assert_size_stride(arg433_1, (320, ), (1, ))
    assert_size_stride(arg434_1, (320, 320), (320, 1))
    assert_size_stride(arg435_1, (320, ), (1, ))
    assert_size_stride(arg436_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg437_1, (320, ), (1, ))
    assert_size_stride(arg438_1, (320, ), (1, ))
    assert_size_stride(arg439_1, (320, ), (1, ))
    assert_size_stride(arg440_1, (640, 320), (320, 1))
    assert_size_stride(arg441_1, (640, ), (1, ))
    assert_size_stride(arg442_1, (320, 320), (320, 1))
    assert_size_stride(arg443_1, (320, ), (1, ))
    assert_size_stride(arg444_1, (320, ), (1, ))
    assert_size_stride(arg445_1, (320, ), (1, ))
    assert_size_stride(arg446_1, (1280, 320), (320, 1))
    assert_size_stride(arg447_1, (1280, ), (1, ))
    assert_size_stride(arg448_1, (320, 1280), (1280, 1))
    assert_size_stride(arg449_1, (320, ), (1, ))
    assert_size_stride(arg450_1, (320, ), (1, ))
    assert_size_stride(arg451_1, (320, ), (1, ))
    assert_size_stride(arg452_1, (320, 320), (320, 1))
    assert_size_stride(arg453_1, (320, ), (1, ))
    assert_size_stride(arg454_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg455_1, (320, ), (1, ))
    assert_size_stride(arg456_1, (320, ), (1, ))
    assert_size_stride(arg457_1, (320, ), (1, ))
    assert_size_stride(arg458_1, (640, 320), (320, 1))
    assert_size_stride(arg459_1, (640, ), (1, ))
    assert_size_stride(arg460_1, (320, 320), (320, 1))
    assert_size_stride(arg461_1, (320, ), (1, ))
    assert_size_stride(arg462_1, (320, ), (1, ))
    assert_size_stride(arg463_1, (320, ), (1, ))
    assert_size_stride(arg464_1, (1280, 320), (320, 1))
    assert_size_stride(arg465_1, (1280, ), (1, ))
    assert_size_stride(arg466_1, (320, 1280), (1280, 1))
    assert_size_stride(arg467_1, (320, ), (1, ))
    assert_size_stride(arg468_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg469_1, (512, ), (1, ))
    assert_size_stride(arg470_1, (512, ), (1, ))
    assert_size_stride(arg471_1, (512, ), (1, ))
    assert_size_stride(arg472_1, (512, ), (1, ))
    assert_size_stride(arg473_1, (512, ), (1, ))
    assert_size_stride(arg474_1, (512, 512), (512, 1))
    assert_size_stride(arg475_1, (512, ), (1, ))
    assert_size_stride(arg476_1, (1024, 512), (512, 1))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (512, 512), (512, 1))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (512, ), (1, ))
    assert_size_stride(arg482_1, (2048, 512), (512, 1))
    assert_size_stride(arg483_1, (2048, ), (1, ))
    assert_size_stride(arg484_1, (512, 2048), (2048, 1))
    assert_size_stride(arg485_1, (512, ), (1, ))
    assert_size_stride(arg486_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg487_1, (512, ), (1, ))
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (512, ), (1, ))
    assert_size_stride(arg490_1, (512, 512), (512, 1))
    assert_size_stride(arg491_1, (512, ), (1, ))
    assert_size_stride(arg492_1, (1024, 512), (512, 1))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (512, 512), (512, 1))
    assert_size_stride(arg495_1, (512, ), (1, ))
    assert_size_stride(arg496_1, (512, ), (1, ))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (2048, 512), (512, 1))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (512, 2048), (2048, 1))
    assert_size_stride(arg501_1, (512, ), (1, ))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (512, 512), (512, 1))
    assert_size_stride(arg505_1, (512, ), (1, ))
    assert_size_stride(arg506_1, (1024, 512), (512, 1))
    assert_size_stride(arg507_1, (1024, ), (1, ))
    assert_size_stride(arg508_1, (512, 512), (512, 1))
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (512, ), (1, ))
    assert_size_stride(arg512_1, (2048, 512), (512, 1))
    assert_size_stride(arg513_1, (2048, ), (1, ))
    assert_size_stride(arg514_1, (512, 2048), (2048, 1))
    assert_size_stride(arg515_1, (512, ), (1, ))
    assert_size_stride(arg516_1, (512, ), (1, ))
    assert_size_stride(arg517_1, (512, ), (1, ))
    assert_size_stride(arg518_1, (1000, 512), (512, 1))
    assert_size_stride(arg519_1, (1000, ), (1, ))
    assert_size_stride(arg520_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg520_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg520_1
        buf1 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 16, grid=grid(192, 16), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf2, arg1_1, buf3, buf4, 25088, 64, grid=grid(25088), stream=stream0)
        buf6 = empty_strided((8, 3136, 64), (200704, 1, 3136), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_3.run(buf2, arg1_1, buf3, buf4, arg2_1, arg3_1, buf6, 1605632, grid=grid(1605632), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        del buf3
        buf10 = reinterpret_tensor(buf2, (8, 3136, 64), (200704, 64, 1), 0); del buf2  # reuse
        # Source Nodes: [l__mod___blocks_0_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf6, arg4_1, arg5_1, buf10, 25088, 64, grid=grid(25088), stream=stream0)
        del arg4_1
        del arg5_1
        buf11 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg8_1, buf11, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg8_1
        # Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf10, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf11, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf16 = reinterpret_tensor(buf4, (8, 49, 64), (3136, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf12, arg9_1, arg10_1, arg11_1, buf16, 392, 64, grid=grid(392), stream=stream0)
        del arg10_1
        del arg11_1
        del arg9_1
        del buf12
        buf17 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf16, (392, 64), (64, 1), 0), reinterpret_tensor(arg12_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf17)
        del arg12_1
        del arg13_1
        buf18 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf10, (25088, 64), (64, 1), 0), reinterpret_tensor(arg6_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf18)
        del arg6_1
        del arg7_1
        del buf10
        # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf19 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf18, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf20 = buf19[0]
        del buf19
        buf24 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (25088, 64), (64, 1), 0), reinterpret_tensor(arg14_1, (64, 64), (1, 64), 0), out=buf24)
        del arg14_1
        buf28 = reinterpret_tensor(buf20, (8, 3136, 64), (200704, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [l__mod___blocks_0_0_norm2, x_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf6, buf24, arg15_1, arg16_1, arg17_1, buf28, 25088, 64, grid=grid(25088), stream=stream0)
        del arg16_1
        del arg17_1
        buf29 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (25088, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 512), (1, 64), 0), out=buf29)
        del arg18_1
        buf30 = reinterpret_tensor(buf29, (8, 3136, 512), (1605632, 512, 1), 0); del buf29  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf30, arg19_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg19_1
        buf31 = reinterpret_tensor(buf28, (25088, 64), (64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (25088, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 64), (1, 512), 0), out=buf31)
        del arg20_1
        buf32 = reinterpret_tensor(buf31, (8, 3136, 64), (200704, 64, 1), 0); del buf31  # reuse
        # Source Nodes: [x_11, x_19], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf32, buf6, buf24, arg15_1, arg21_1, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del arg15_1
        del arg21_1
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(reinterpret_tensor(buf32, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), arg22_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf33, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg22_1
        buf37 = reinterpret_tensor(buf6, (8, 3136, 64), (200704, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [l__mod___blocks_0_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_10.run(buf33, arg23_1, buf32, arg24_1, arg25_1, buf37, 25088, 64, grid=grid(25088), stream=stream0)
        del arg24_1
        del arg25_1
        buf38 = buf11; del buf11  # reuse
        # Source Nodes: [l__mod___blocks_0_1_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg28_1, buf38, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg28_1
        # Source Nodes: [l__mod___blocks_0_1_attn_sr], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(reinterpret_tensor(buf37, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf38, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf43 = buf16; del buf16  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf39, arg29_1, arg30_1, arg31_1, buf43, 392, 64, grid=grid(392), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        del buf39
        buf44 = buf17; del buf17  # reuse
        # Source Nodes: [l__mod___blocks_0_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg33_1, reinterpret_tensor(buf43, (392, 64), (64, 1), 0), reinterpret_tensor(arg32_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf44)
        del arg32_1
        del arg33_1
        buf45 = buf24; del buf24  # reuse
        # Source Nodes: [l__mod___blocks_0_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf37, (25088, 64), (64, 1), 0), reinterpret_tensor(arg26_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf45)
        del arg26_1
        del arg27_1
        del buf37
        # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf46 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf45, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf44, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf44, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf47 = buf46[0]
        del buf46
        buf51 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (25088, 64), (64, 1), 0), reinterpret_tensor(arg34_1, (64, 64), (1, 64), 0), out=buf51)
        del arg34_1
        buf52 = reinterpret_tensor(buf51, (8, 3136, 64), (200704, 64, 1), 0); del buf51  # reuse
        buf56 = reinterpret_tensor(buf47, (8, 3136, 64), (200704, 64, 1), 0); del buf47  # reuse
        # Source Nodes: [l__mod___blocks_0_1_norm2, x_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf52, buf33, arg23_1, buf32, arg35_1, arg36_1, arg37_1, buf56, 25088, 64, grid=grid(25088), stream=stream0)
        del arg23_1
        del arg35_1
        del arg36_1
        del arg37_1
        buf57 = reinterpret_tensor(buf30, (25088, 512), (512, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (25088, 64), (64, 1), 0), reinterpret_tensor(arg38_1, (64, 512), (1, 64), 0), out=buf57)
        del arg38_1
        buf58 = reinterpret_tensor(buf57, (8, 3136, 512), (1605632, 512, 1), 0); del buf57  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf58, arg39_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg39_1
        buf59 = reinterpret_tensor(buf56, (25088, 64), (64, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (25088, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 64), (1, 512), 0), out=buf59)
        del arg40_1
        buf63 = reinterpret_tensor(buf33, (8, 3136, 64), (200704, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [l__mod___blocks_0_2_norm1, x_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf52, buf59, arg41_1, arg42_1, arg43_1, buf63, 25088, 64, grid=grid(25088), stream=stream0)
        del arg42_1
        del arg43_1
        buf64 = buf38; del buf38  # reuse
        # Source Nodes: [l__mod___blocks_0_2_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg46_1, buf64, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg46_1
        # Source Nodes: [l__mod___blocks_0_2_attn_sr], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(reinterpret_tensor(buf63, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf64, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 64, 7, 7), (3136, 49, 7, 1))
        buf69 = buf43; del buf43  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf65, arg47_1, arg48_1, arg49_1, buf69, 392, 64, grid=grid(392), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del buf65
        buf70 = buf44; del buf44  # reuse
        # Source Nodes: [l__mod___blocks_0_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg51_1, reinterpret_tensor(buf69, (392, 64), (64, 1), 0), reinterpret_tensor(arg50_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf70)
        del arg50_1
        del arg51_1
        del buf69
        buf71 = reinterpret_tensor(buf32, (25088, 64), (64, 1), 0); del buf32  # reuse
        # Source Nodes: [l__mod___blocks_0_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf63, (25088, 64), (64, 1), 0), reinterpret_tensor(arg44_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf71)
        del arg44_1
        del arg45_1
        del buf63
        # Source Nodes: [x_43], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf72 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf71, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf70, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf70, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf73 = buf72[0]
        del buf72
        buf77 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (25088, 64), (64, 1), 0), reinterpret_tensor(arg52_1, (64, 64), (1, 64), 0), out=buf77)
        del arg52_1
        buf78 = reinterpret_tensor(buf77, (8, 3136, 64), (200704, 64, 1), 0); del buf77  # reuse
        buf82 = reinterpret_tensor(buf73, (8, 3136, 64), (200704, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [l__mod___blocks_0_2_norm2, x_39, x_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf78, buf52, buf59, arg41_1, arg53_1, arg54_1, arg55_1, buf82, 25088, 64, grid=grid(25088), stream=stream0)
        del arg41_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf52
        del buf59
        buf83 = reinterpret_tensor(buf58, (25088, 512), (512, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (25088, 64), (64, 1), 0), reinterpret_tensor(arg56_1, (64, 512), (1, 64), 0), out=buf83)
        del arg56_1
        buf84 = reinterpret_tensor(buf83, (8, 3136, 512), (1605632, 512, 1), 0); del buf83  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf84, arg57_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg57_1
        buf85 = reinterpret_tensor(buf82, (25088, 64), (64, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (25088, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 64), (1, 512), 0), out=buf85)
        del arg58_1
        del buf84
        buf86 = reinterpret_tensor(buf85, (8, 3136, 64), (200704, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf86, buf78, arg59_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg59_1
        del buf78
        buf87 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embeds_1_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg60_1, buf87, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del arg60_1
        # Source Nodes: [l__mod___patch_embeds_1_proj], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(reinterpret_tensor(buf86, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf87, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 128, 28, 28), (100352, 784, 28, 1))
        del buf86
        del buf87
        buf89 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_16.run(buf88, arg61_1, buf89, buf90, 6272, 128, grid=grid(6272), stream=stream0)
        buf92 = empty_strided((8, 784, 128), (100352, 1, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf88, arg61_1, buf89, buf90, arg62_1, arg63_1, buf92, 802816, grid=grid(802816), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del buf89
        del buf90
        buf96 = reinterpret_tensor(buf88, (8, 784, 128), (100352, 1, 784), 0); del buf88  # reuse
        # Source Nodes: [l__mod___blocks_1_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_18.run(buf92, arg64_1, arg65_1, buf96, 6272, 128, grid=grid(6272), stream=stream0)
        del arg64_1
        del arg65_1
        buf97 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf96, buf97, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf98 = reinterpret_tensor(buf64, (128, 128, 4, 4), (2048, 1, 512, 128), 0); del buf64  # reuse
        # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg68_1, buf98, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg68_1
        # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf97, buf98, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf103 = reinterpret_tensor(buf70, (8, 49, 128), (6272, 128, 1), 0); del buf70  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_21.run(buf99, arg69_1, arg70_1, arg71_1, buf103, 392, 128, grid=grid(392), stream=stream0)
        del arg69_1
        del arg70_1
        del arg71_1
        del buf99
        buf104 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf103, (392, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf104)
        del arg72_1
        del arg73_1
        buf105 = reinterpret_tensor(buf97, (6272, 128), (1, 6272), 0); del buf97  # reuse
        # Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
        triton_poi_fused_addmm_22.run(buf96, buf105, 802816, grid=grid(802816), stream=stream0)
        buf106 = reinterpret_tensor(buf96, (6272, 128), (128, 1), 0); del buf96  # reuse
        # Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg67_1, buf105, reinterpret_tensor(arg66_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf106)
        del arg66_1
        del arg67_1
        del buf105
        # Source Nodes: [x_64], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf107 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf106, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf104, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf104, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf108 = buf107[0]
        del buf107
        buf112 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (6272, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 128), (1, 128), 0), out=buf112)
        del arg74_1
        buf116 = reinterpret_tensor(buf108, (8, 784, 128), (100352, 128, 1), 0); del buf108  # reuse
        # Source Nodes: [l__mod___blocks_1_0_norm2, x_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_23.run(buf92, buf112, arg75_1, arg76_1, arg77_1, buf116, 6272, 128, grid=grid(6272), stream=stream0)
        del arg76_1
        del arg77_1
        buf117 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (6272, 128), (128, 1), 0), reinterpret_tensor(arg78_1, (128, 1024), (1, 128), 0), out=buf117)
        del arg78_1
        buf118 = reinterpret_tensor(buf117, (8, 784, 1024), (802816, 1024, 1), 0); del buf117  # reuse
        # Source Nodes: [x_70], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf118, arg79_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg79_1
        buf119 = reinterpret_tensor(buf116, (6272, 128), (128, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 128), (1, 1024), 0), out=buf119)
        del arg80_1
        buf120 = reinterpret_tensor(buf119, (8, 784, 128), (100352, 128, 1), 0); del buf119  # reuse
        # Source Nodes: [x_68, x_76], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf120, buf92, buf112, arg75_1, arg81_1, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del arg75_1
        del arg81_1
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(reinterpret_tensor(buf120, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), arg82_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf121, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg82_1
        buf125 = reinterpret_tensor(buf92, (8, 784, 128), (100352, 128, 1), 0); del buf92  # reuse
        # Source Nodes: [l__mod___blocks_1_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_26.run(buf121, arg83_1, buf120, arg84_1, arg85_1, buf125, 6272, 128, grid=grid(6272), stream=stream0)
        del arg84_1
        del arg85_1
        buf126 = buf98; del buf98  # reuse
        # Source Nodes: [l__mod___blocks_1_1_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg88_1, buf126, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg88_1
        # Source Nodes: [l__mod___blocks_1_1_attn_sr], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(reinterpret_tensor(buf125, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf126, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf131 = buf103; del buf103  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_21.run(buf127, arg89_1, arg90_1, arg91_1, buf131, 392, 128, grid=grid(392), stream=stream0)
        del arg89_1
        del arg90_1
        del arg91_1
        del buf127
        buf132 = buf104; del buf104  # reuse
        # Source Nodes: [l__mod___blocks_1_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf131, (392, 128), (128, 1), 0), reinterpret_tensor(arg92_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf132)
        del arg92_1
        del arg93_1
        buf133 = buf112; del buf112  # reuse
        # Source Nodes: [l__mod___blocks_1_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg87_1, reinterpret_tensor(buf125, (6272, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf133)
        del arg86_1
        del arg87_1
        del buf125
        # Source Nodes: [x_84], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf134 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf133, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf132, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf132, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf135 = buf134[0]
        del buf134
        buf139 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (6272, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 128), (1, 128), 0), out=buf139)
        del arg94_1
        buf140 = reinterpret_tensor(buf139, (8, 784, 128), (100352, 128, 1), 0); del buf139  # reuse
        buf144 = reinterpret_tensor(buf135, (8, 784, 128), (100352, 128, 1), 0); del buf135  # reuse
        # Source Nodes: [l__mod___blocks_1_1_norm2, x_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf140, buf121, arg83_1, buf120, arg95_1, arg96_1, arg97_1, buf144, 6272, 128, grid=grid(6272), stream=stream0)
        del arg83_1
        del arg95_1
        del arg96_1
        del arg97_1
        buf145 = reinterpret_tensor(buf118, (6272, 1024), (1024, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (6272, 128), (128, 1), 0), reinterpret_tensor(arg98_1, (128, 1024), (1, 128), 0), out=buf145)
        del arg98_1
        buf146 = reinterpret_tensor(buf145, (8, 784, 1024), (802816, 1024, 1), 0); del buf145  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf146, arg99_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg99_1
        buf147 = reinterpret_tensor(buf144, (6272, 128), (128, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg100_1, (1024, 128), (1, 1024), 0), out=buf147)
        del arg100_1
        buf151 = reinterpret_tensor(buf121, (8, 784, 128), (100352, 128, 1), 0); del buf121  # reuse
        # Source Nodes: [l__mod___blocks_1_2_norm1, x_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf140, buf147, arg101_1, arg102_1, arg103_1, buf151, 6272, 128, grid=grid(6272), stream=stream0)
        del arg102_1
        del arg103_1
        buf152 = buf126; del buf126  # reuse
        # Source Nodes: [l__mod___blocks_1_2_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg106_1, buf152, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg106_1
        # Source Nodes: [l__mod___blocks_1_2_attn_sr], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(reinterpret_tensor(buf151, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf152, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 128, 7, 7), (6272, 49, 7, 1))
        buf157 = buf131; del buf131  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_21.run(buf153, arg107_1, arg108_1, arg109_1, buf157, 392, 128, grid=grid(392), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del buf153
        buf158 = buf132; del buf132  # reuse
        # Source Nodes: [l__mod___blocks_1_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf157, (392, 128), (128, 1), 0), reinterpret_tensor(arg110_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf158)
        del arg110_1
        del arg111_1
        buf159 = reinterpret_tensor(buf120, (6272, 128), (128, 1), 0); del buf120  # reuse
        # Source Nodes: [l__mod___blocks_1_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf151, (6272, 128), (128, 1), 0), reinterpret_tensor(arg104_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf159)
        del arg104_1
        del arg105_1
        del buf151
        # Source Nodes: [x_100], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf160 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf159, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf158, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf158, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf161 = buf160[0]
        del buf160
        buf165 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (6272, 128), (128, 1), 0), reinterpret_tensor(arg112_1, (128, 128), (1, 128), 0), out=buf165)
        del arg112_1
        buf166 = reinterpret_tensor(buf165, (8, 784, 128), (100352, 128, 1), 0); del buf165  # reuse
        buf170 = reinterpret_tensor(buf161, (8, 784, 128), (100352, 128, 1), 0); del buf161  # reuse
        # Source Nodes: [l__mod___blocks_1_2_norm2, x_104, x_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf166, buf140, buf147, arg101_1, arg113_1, arg114_1, arg115_1, buf170, 6272, 128, grid=grid(6272), stream=stream0)
        del arg101_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf171 = reinterpret_tensor(buf146, (6272, 1024), (1024, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (6272, 128), (128, 1), 0), reinterpret_tensor(arg116_1, (128, 1024), (1, 128), 0), out=buf171)
        del arg116_1
        buf172 = reinterpret_tensor(buf171, (8, 784, 1024), (802816, 1024, 1), 0); del buf171  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf172, arg117_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg117_1
        buf173 = reinterpret_tensor(buf170, (6272, 128), (128, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 128), (1, 1024), 0), out=buf173)
        del arg118_1
        buf177 = reinterpret_tensor(buf147, (8, 784, 128), (100352, 128, 1), 0); del buf147  # reuse
        # Source Nodes: [l__mod___blocks_1_3_norm1, x_112], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf166, buf173, arg119_1, arg120_1, arg121_1, buf177, 6272, 128, grid=grid(6272), stream=stream0)
        del arg120_1
        del arg121_1
        buf178 = buf152; del buf152  # reuse
        # Source Nodes: [l__mod___blocks_1_3_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg124_1, buf178, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg124_1
        # Source Nodes: [l__mod___blocks_1_3_attn_sr], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(reinterpret_tensor(buf177, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf178, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 128, 7, 7), (6272, 49, 7, 1))
        del buf178
        buf183 = buf157; del buf157  # reuse
        # Source Nodes: [x_115], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_21.run(buf179, arg125_1, arg126_1, arg127_1, buf183, 392, 128, grid=grid(392), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        del buf179
        buf184 = buf158; del buf158  # reuse
        # Source Nodes: [l__mod___blocks_1_3_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg129_1, reinterpret_tensor(buf183, (392, 128), (128, 1), 0), reinterpret_tensor(arg128_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf184)
        del arg128_1
        del arg129_1
        del buf183
        buf185 = reinterpret_tensor(buf140, (6272, 128), (128, 1), 0); del buf140  # reuse
        # Source Nodes: [l__mod___blocks_1_3_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf177, (6272, 128), (128, 1), 0), reinterpret_tensor(arg122_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf185)
        del arg122_1
        del arg123_1
        del buf177
        # Source Nodes: [x_116], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf186 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf185, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf184, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf184, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        del buf184
        buf187 = buf186[0]
        del buf186
        buf191 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (6272, 128), (128, 1), 0), reinterpret_tensor(arg130_1, (128, 128), (1, 128), 0), out=buf191)
        del arg130_1
        buf192 = reinterpret_tensor(buf191, (8, 784, 128), (100352, 128, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf187, (8, 784, 128), (100352, 128, 1), 0); del buf187  # reuse
        # Source Nodes: [l__mod___blocks_1_3_norm2, x_112, x_120], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf192, buf166, buf173, arg119_1, arg131_1, arg132_1, arg133_1, buf196, 6272, 128, grid=grid(6272), stream=stream0)
        del arg119_1
        del arg131_1
        del arg132_1
        del arg133_1
        del buf166
        del buf173
        buf197 = reinterpret_tensor(buf172, (6272, 1024), (1024, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (6272, 128), (128, 1), 0), reinterpret_tensor(arg134_1, (128, 1024), (1, 128), 0), out=buf197)
        del arg134_1
        buf198 = reinterpret_tensor(buf197, (8, 784, 1024), (802816, 1024, 1), 0); del buf197  # reuse
        # Source Nodes: [x_122], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf198, arg135_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg135_1
        buf199 = reinterpret_tensor(buf196, (6272, 128), (128, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 128), (1, 1024), 0), out=buf199)
        del arg136_1
        del buf198
        buf200 = reinterpret_tensor(buf199, (8, 784, 128), (100352, 128, 1), 0); del buf199  # reuse
        # Source Nodes: [x_128], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf200, buf192, arg137_1, 802816, grid=grid(802816), stream=stream0)
        del arg137_1
        del buf192
        buf201 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embeds_2_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(arg138_1, buf201, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del arg138_1
        # Source Nodes: [l__mod___patch_embeds_2_proj], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(reinterpret_tensor(buf200, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf201, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 320, 14, 14), (62720, 196, 14, 1))
        del buf201
        buf203 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf204 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_32.run(buf202, arg139_1, buf203, buf204, buf205, 4704, 107, grid=grid(4704), stream=stream0)
        buf206 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf207 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_33.run(buf203, buf204, buf205, buf206, buf207, 1568, 3, grid=grid(1568), stream=stream0)
        buf209 = reinterpret_tensor(buf202, (8, 196, 320), (62720, 1, 196), 0); del buf202  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_34.run(buf209, arg139_1, buf206, buf207, arg140_1, arg141_1, 501760, grid=grid(501760), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        buf210 = buf205; del buf205  # reuse
        buf211 = buf204; del buf204  # reuse
        buf212 = buf203; del buf203  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_35.run(buf209, buf210, buf211, buf212, 4704, 107, grid=grid(4704), stream=stream0)
        buf213 = buf207; del buf207  # reuse
        buf214 = buf206; del buf206  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_33.run(buf210, buf211, buf212, buf213, buf214, 1568, 3, grid=grid(1568), stream=stream0)
        buf216 = empty_strided((8, 196, 320), (62720, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_36.run(buf209, buf213, buf214, arg142_1, arg143_1, buf216, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del arg142_1
        del arg143_1
        buf217 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf216, buf217, 2560, 196, grid=grid(2560, 196), stream=stream0)
        buf218 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg146_1, buf218, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg146_1
        # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf217, buf218, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf220 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        buf221 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        buf222 = empty_strided((8, 49, 1, 3), (147, 1, 1176, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf219, arg147_1, buf220, buf221, buf222, 1176, 107, grid=grid(1176), stream=stream0)
        buf223 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf224 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf220, buf221, buf222, buf223, buf224, 392, 3, grid=grid(392), stream=stream0)
        buf226 = empty((8, 49, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf219, arg147_1, buf223, buf224, arg148_1, arg149_1, buf226, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del buf219
        buf227 = empty((392, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg151_1, reinterpret_tensor(buf226, (392, 320), (320, 1), 0), reinterpret_tensor(arg150_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf227)
        del arg150_1
        del arg151_1
        buf228 = reinterpret_tensor(buf217, (1568, 320), (1, 1568), 0); del buf217  # reuse
        # Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
        triton_poi_fused_addmm_42.run(buf216, buf228, 501760, grid=grid(501760), stream=stream0)
        buf229 = reinterpret_tensor(buf216, (1568, 320), (320, 1), 0); del buf216  # reuse
        # Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg145_1, buf228, reinterpret_tensor(arg144_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf229)
        del arg144_1
        del arg145_1
        del buf228
        # Source Nodes: [x_137], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf230 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf229, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf227, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf227, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf231 = buf230[0]
        del buf230
        buf235 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1568, 320), (320, 1), 0), reinterpret_tensor(arg152_1, (320, 320), (1, 320), 0), out=buf235)
        del arg152_1
        buf236 = reinterpret_tensor(buf212, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf212  # reuse
        buf237 = reinterpret_tensor(buf211, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf211  # reuse
        buf238 = reinterpret_tensor(buf210, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf210  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_43.run(buf209, buf235, arg153_1, buf236, buf237, buf238, 4704, 107, grid=grid(4704), stream=stream0)
        buf239 = buf214; del buf214  # reuse
        buf240 = buf213; del buf213  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_44.run(buf236, buf237, buf238, buf239, buf240, 1568, 3, grid=grid(1568), stream=stream0)
        buf242 = reinterpret_tensor(buf231, (8, 196, 320), (62720, 320, 1), 0); del buf231  # reuse
        # Source Nodes: [l__mod___blocks_2_0_norm2, x_141], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_45.run(buf209, buf235, arg153_1, buf239, buf240, arg154_1, arg155_1, buf242, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del arg154_1
        del arg155_1
        buf243 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 320), (320, 1), 0), reinterpret_tensor(arg156_1, (320, 1280), (1, 320), 0), out=buf243)
        del arg156_1
        buf244 = reinterpret_tensor(buf243, (8, 196, 1280), (250880, 1280, 1), 0); del buf243  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf244, arg157_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg157_1
        buf245 = reinterpret_tensor(buf242, (1568, 320), (320, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg158_1, (1280, 320), (1, 1280), 0), out=buf245)
        del arg158_1
        buf246 = reinterpret_tensor(buf235, (8, 196, 320), (62720, 320, 1), 0); del buf235  # reuse
        # Source Nodes: [x_141, x_149], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(buf246, buf209, arg153_1, buf245, arg159_1, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del arg153_1
        del arg159_1
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(reinterpret_tensor(buf246, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), arg160_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf247, (8, 320, 14, 14), (62720, 196, 14, 1))
        del arg160_1
        buf248 = buf238; del buf238  # reuse
        buf249 = buf237; del buf237  # reuse
        buf250 = buf236; del buf236  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_48.run(buf247, arg161_1, buf246, buf248, buf249, buf250, 4704, 107, grid=grid(4704), stream=stream0)
        buf251 = buf240; del buf240  # reuse
        buf252 = buf239; del buf239  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_44.run(buf248, buf249, buf250, buf251, buf252, 1568, 3, grid=grid(1568), stream=stream0)
        del buf248
        del buf249
        del buf250
        buf254 = reinterpret_tensor(buf245, (8, 196, 320), (62720, 320, 1), 0); del buf245  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_49.run(buf247, arg161_1, buf246, buf251, buf252, arg162_1, arg163_1, buf254, 1568, 320, grid=grid(1568, 320), stream=stream0)
        del arg162_1
        del arg163_1
        buf255 = buf218; del buf218  # reuse
        # Source Nodes: [l__mod___blocks_2_1_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg166_1, buf255, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg166_1
        # Source Nodes: [l__mod___blocks_2_1_attn_sr], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(reinterpret_tensor(buf254, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf255, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf257 = buf222; del buf222  # reuse
        buf258 = buf221; del buf221  # reuse
        buf259 = buf220; del buf220  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf256, arg167_1, buf257, buf258, buf259, 1176, 107, grid=grid(1176), stream=stream0)
        buf260 = buf224; del buf224  # reuse
        buf261 = buf223; del buf223  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf257, buf258, buf259, buf260, buf261, 392, 3, grid=grid(392), stream=stream0)
        buf263 = buf226; del buf226  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf256, arg167_1, buf260, buf261, arg168_1, arg169_1, buf263, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del buf256
        buf264 = buf227; del buf227  # reuse
        # Source Nodes: [l__mod___blocks_2_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf263, (392, 320), (320, 1), 0), reinterpret_tensor(arg170_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf264)
        del arg170_1
        del arg171_1
        buf265 = reinterpret_tensor(buf209, (1568, 320), (320, 1), 0); del buf209  # reuse
        # Source Nodes: [l__mod___blocks_2_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg165_1, reinterpret_tensor(buf254, (1568, 320), (320, 1), 0), reinterpret_tensor(arg164_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf265)
        del arg164_1
        del arg165_1
        del buf254
        # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf266 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf265, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf264, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf264, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf267 = buf266[0]
        del buf266
        buf271 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1568, 320), (320, 1), 0), reinterpret_tensor(arg172_1, (320, 320), (1, 320), 0), out=buf271)
        del arg172_1
        buf272 = buf246; del buf246  # reuse
        buf276 = reinterpret_tensor(buf267, (8, 196, 320), (62720, 320, 1), 0); del buf267  # reuse
        # Source Nodes: [l__mod___blocks_2_1_norm2, x_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_50.run(buf272, buf247, arg161_1, buf271, arg173_1, arg174_1, arg175_1, buf276, 1568, 320, grid=grid(1568), stream=stream0)
        del arg161_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf277 = reinterpret_tensor(buf244, (1568, 1280), (1280, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1568, 320), (320, 1), 0), reinterpret_tensor(arg176_1, (320, 1280), (1, 320), 0), out=buf277)
        del arg176_1
        buf278 = reinterpret_tensor(buf277, (8, 196, 1280), (250880, 1280, 1), 0); del buf277  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf278, arg177_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg177_1
        buf279 = reinterpret_tensor(buf276, (1568, 320), (320, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg178_1, (1280, 320), (1, 1280), 0), out=buf279)
        del arg178_1
        buf283 = reinterpret_tensor(buf271, (8, 196, 320), (62720, 320, 1), 0); del buf271  # reuse
        # Source Nodes: [l__mod___blocks_2_2_norm1, x_169], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf272, buf279, arg179_1, arg180_1, arg181_1, buf283, 1568, 320, grid=grid(1568), stream=stream0)
        del arg180_1
        del arg181_1
        buf284 = buf255; del buf255  # reuse
        # Source Nodes: [l__mod___blocks_2_2_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg184_1, buf284, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg184_1
        # Source Nodes: [l__mod___blocks_2_2_attn_sr], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(reinterpret_tensor(buf283, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf284, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf286 = buf259; del buf259  # reuse
        buf287 = buf258; del buf258  # reuse
        buf288 = buf257; del buf257  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf285, arg185_1, buf286, buf287, buf288, 1176, 107, grid=grid(1176), stream=stream0)
        buf289 = buf261; del buf261  # reuse
        buf290 = buf260; del buf260  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf286, buf287, buf288, buf289, buf290, 392, 3, grid=grid(392), stream=stream0)
        buf292 = buf263; del buf263  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf285, arg185_1, buf289, buf290, arg186_1, arg187_1, buf292, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg185_1
        del arg186_1
        del arg187_1
        del buf285
        buf293 = buf264; del buf264  # reuse
        # Source Nodes: [l__mod___blocks_2_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg189_1, reinterpret_tensor(buf292, (392, 320), (320, 1), 0), reinterpret_tensor(arg188_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf293)
        del arg188_1
        del arg189_1
        buf294 = reinterpret_tensor(buf247, (1568, 320), (320, 1), 0); del buf247  # reuse
        # Source Nodes: [l__mod___blocks_2_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg183_1, reinterpret_tensor(buf283, (1568, 320), (320, 1), 0), reinterpret_tensor(arg182_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf294)
        del arg182_1
        del arg183_1
        del buf283
        # Source Nodes: [x_173], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf295 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf294, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf293, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf293, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf296 = buf295[0]
        del buf295
        buf300 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (1568, 320), (320, 1), 0), reinterpret_tensor(arg190_1, (320, 320), (1, 320), 0), out=buf300)
        del arg190_1
        buf301 = reinterpret_tensor(buf300, (8, 196, 320), (62720, 320, 1), 0); del buf300  # reuse
        buf305 = reinterpret_tensor(buf296, (8, 196, 320), (62720, 320, 1), 0); del buf296  # reuse
        # Source Nodes: [l__mod___blocks_2_2_norm2, x_169, x_177], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf301, buf272, buf279, arg179_1, arg191_1, arg192_1, arg193_1, buf305, 1568, 320, grid=grid(1568), stream=stream0)
        del arg179_1
        del arg191_1
        del arg192_1
        del arg193_1
        buf306 = reinterpret_tensor(buf278, (1568, 1280), (1280, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf305, (1568, 320), (320, 1), 0), reinterpret_tensor(arg194_1, (320, 1280), (1, 320), 0), out=buf306)
        del arg194_1
        buf307 = reinterpret_tensor(buf306, (8, 196, 1280), (250880, 1280, 1), 0); del buf306  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf307, arg195_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg195_1
        buf308 = reinterpret_tensor(buf305, (1568, 320), (320, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg196_1, (1280, 320), (1, 1280), 0), out=buf308)
        del arg196_1
        buf312 = reinterpret_tensor(buf279, (8, 196, 320), (62720, 320, 1), 0); del buf279  # reuse
        # Source Nodes: [l__mod___blocks_2_3_norm1, x_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf301, buf308, arg197_1, arg198_1, arg199_1, buf312, 1568, 320, grid=grid(1568), stream=stream0)
        del arg198_1
        del arg199_1
        buf313 = buf284; del buf284  # reuse
        # Source Nodes: [l__mod___blocks_2_3_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg202_1, buf313, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg202_1
        # Source Nodes: [l__mod___blocks_2_3_attn_sr], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(reinterpret_tensor(buf312, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf313, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf315 = buf288; del buf288  # reuse
        buf316 = buf287; del buf287  # reuse
        buf317 = buf286; del buf286  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf314, arg203_1, buf315, buf316, buf317, 1176, 107, grid=grid(1176), stream=stream0)
        buf318 = buf290; del buf290  # reuse
        buf319 = buf289; del buf289  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf315, buf316, buf317, buf318, buf319, 392, 3, grid=grid(392), stream=stream0)
        buf321 = buf292; del buf292  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf314, arg203_1, buf318, buf319, arg204_1, arg205_1, buf321, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del buf314
        buf322 = buf293; del buf293  # reuse
        # Source Nodes: [l__mod___blocks_2_3_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg207_1, reinterpret_tensor(buf321, (392, 320), (320, 1), 0), reinterpret_tensor(arg206_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf322)
        del arg206_1
        del arg207_1
        buf323 = reinterpret_tensor(buf272, (1568, 320), (320, 1), 0); del buf272  # reuse
        # Source Nodes: [l__mod___blocks_2_3_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg201_1, reinterpret_tensor(buf312, (1568, 320), (320, 1), 0), reinterpret_tensor(arg200_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf323)
        del arg200_1
        del arg201_1
        del buf312
        # Source Nodes: [x_189], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf324 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf323, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf325 = buf324[0]
        del buf324
        buf329 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf325, (1568, 320), (320, 1), 0), reinterpret_tensor(arg208_1, (320, 320), (1, 320), 0), out=buf329)
        del arg208_1
        buf330 = reinterpret_tensor(buf329, (8, 196, 320), (62720, 320, 1), 0); del buf329  # reuse
        buf334 = reinterpret_tensor(buf325, (8, 196, 320), (62720, 320, 1), 0); del buf325  # reuse
        # Source Nodes: [l__mod___blocks_2_3_norm2, x_185, x_193], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf330, buf301, buf308, arg197_1, arg209_1, arg210_1, arg211_1, buf334, 1568, 320, grid=grid(1568), stream=stream0)
        del arg197_1
        del arg209_1
        del arg210_1
        del arg211_1
        buf335 = reinterpret_tensor(buf307, (1568, 1280), (1280, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1568, 320), (320, 1), 0), reinterpret_tensor(arg212_1, (320, 1280), (1, 320), 0), out=buf335)
        del arg212_1
        buf336 = reinterpret_tensor(buf335, (8, 196, 1280), (250880, 1280, 1), 0); del buf335  # reuse
        # Source Nodes: [x_195], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf336, arg213_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg213_1
        buf337 = reinterpret_tensor(buf334, (1568, 320), (320, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg214_1, (1280, 320), (1, 1280), 0), out=buf337)
        del arg214_1
        buf341 = reinterpret_tensor(buf308, (8, 196, 320), (62720, 320, 1), 0); del buf308  # reuse
        # Source Nodes: [l__mod___blocks_2_4_norm1, x_201], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf330, buf337, arg215_1, arg216_1, arg217_1, buf341, 1568, 320, grid=grid(1568), stream=stream0)
        del arg216_1
        del arg217_1
        buf342 = buf313; del buf313  # reuse
        # Source Nodes: [l__mod___blocks_2_4_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg220_1, buf342, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg220_1
        # Source Nodes: [l__mod___blocks_2_4_attn_sr], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(reinterpret_tensor(buf341, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf342, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf344 = buf317; del buf317  # reuse
        buf345 = buf316; del buf316  # reuse
        buf346 = buf315; del buf315  # reuse
        # Source Nodes: [x_204], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf343, arg221_1, buf344, buf345, buf346, 1176, 107, grid=grid(1176), stream=stream0)
        buf347 = buf319; del buf319  # reuse
        buf348 = buf318; del buf318  # reuse
        # Source Nodes: [x_204], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf344, buf345, buf346, buf347, buf348, 392, 3, grid=grid(392), stream=stream0)
        buf350 = buf321; del buf321  # reuse
        # Source Nodes: [x_204], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf343, arg221_1, buf347, buf348, arg222_1, arg223_1, buf350, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del buf343
        buf351 = buf322; del buf322  # reuse
        # Source Nodes: [l__mod___blocks_2_4_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg225_1, reinterpret_tensor(buf350, (392, 320), (320, 1), 0), reinterpret_tensor(arg224_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf351)
        del arg224_1
        del arg225_1
        buf352 = reinterpret_tensor(buf301, (1568, 320), (320, 1), 0); del buf301  # reuse
        # Source Nodes: [l__mod___blocks_2_4_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg219_1, reinterpret_tensor(buf341, (1568, 320), (320, 1), 0), reinterpret_tensor(arg218_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf352)
        del arg218_1
        del arg219_1
        del buf341
        # Source Nodes: [x_205], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf353 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf352, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf351, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf351, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf354 = buf353[0]
        del buf353
        buf358 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1568, 320), (320, 1), 0), reinterpret_tensor(arg226_1, (320, 320), (1, 320), 0), out=buf358)
        del arg226_1
        buf359 = reinterpret_tensor(buf358, (8, 196, 320), (62720, 320, 1), 0); del buf358  # reuse
        buf363 = reinterpret_tensor(buf354, (8, 196, 320), (62720, 320, 1), 0); del buf354  # reuse
        # Source Nodes: [l__mod___blocks_2_4_norm2, x_201, x_209], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf359, buf330, buf337, arg215_1, arg227_1, arg228_1, arg229_1, buf363, 1568, 320, grid=grid(1568), stream=stream0)
        del arg215_1
        del arg227_1
        del arg228_1
        del arg229_1
        buf364 = reinterpret_tensor(buf336, (1568, 1280), (1280, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (1568, 320), (320, 1), 0), reinterpret_tensor(arg230_1, (320, 1280), (1, 320), 0), out=buf364)
        del arg230_1
        buf365 = reinterpret_tensor(buf364, (8, 196, 1280), (250880, 1280, 1), 0); del buf364  # reuse
        # Source Nodes: [x_211], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf365, arg231_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg231_1
        buf366 = reinterpret_tensor(buf363, (1568, 320), (320, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg232_1, (1280, 320), (1, 1280), 0), out=buf366)
        del arg232_1
        buf370 = reinterpret_tensor(buf337, (8, 196, 320), (62720, 320, 1), 0); del buf337  # reuse
        # Source Nodes: [l__mod___blocks_2_5_norm1, x_217], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf359, buf366, arg233_1, arg234_1, arg235_1, buf370, 1568, 320, grid=grid(1568), stream=stream0)
        del arg234_1
        del arg235_1
        buf371 = buf342; del buf342  # reuse
        # Source Nodes: [l__mod___blocks_2_5_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg238_1, buf371, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg238_1
        # Source Nodes: [l__mod___blocks_2_5_attn_sr], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(reinterpret_tensor(buf370, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf371, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf373 = buf346; del buf346  # reuse
        buf374 = buf345; del buf345  # reuse
        buf375 = buf344; del buf344  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf372, arg239_1, buf373, buf374, buf375, 1176, 107, grid=grid(1176), stream=stream0)
        buf376 = buf348; del buf348  # reuse
        buf377 = buf347; del buf347  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf373, buf374, buf375, buf376, buf377, 392, 3, grid=grid(392), stream=stream0)
        buf379 = buf350; del buf350  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf372, arg239_1, buf376, buf377, arg240_1, arg241_1, buf379, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg239_1
        del arg240_1
        del arg241_1
        del buf372
        buf380 = buf351; del buf351  # reuse
        # Source Nodes: [l__mod___blocks_2_5_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg243_1, reinterpret_tensor(buf379, (392, 320), (320, 1), 0), reinterpret_tensor(arg242_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf380)
        del arg242_1
        del arg243_1
        buf381 = reinterpret_tensor(buf330, (1568, 320), (320, 1), 0); del buf330  # reuse
        # Source Nodes: [l__mod___blocks_2_5_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg237_1, reinterpret_tensor(buf370, (1568, 320), (320, 1), 0), reinterpret_tensor(arg236_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf381)
        del arg236_1
        del arg237_1
        del buf370
        # Source Nodes: [x_221], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf382 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf381, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf380, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf380, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf383 = buf382[0]
        del buf382
        buf387 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (1568, 320), (320, 1), 0), reinterpret_tensor(arg244_1, (320, 320), (1, 320), 0), out=buf387)
        del arg244_1
        buf388 = reinterpret_tensor(buf387, (8, 196, 320), (62720, 320, 1), 0); del buf387  # reuse
        buf392 = reinterpret_tensor(buf383, (8, 196, 320), (62720, 320, 1), 0); del buf383  # reuse
        # Source Nodes: [l__mod___blocks_2_5_norm2, x_217, x_225], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf388, buf359, buf366, arg233_1, arg245_1, arg246_1, arg247_1, buf392, 1568, 320, grid=grid(1568), stream=stream0)
        del arg233_1
        del arg245_1
        del arg246_1
        del arg247_1
        buf393 = reinterpret_tensor(buf365, (1568, 1280), (1280, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (1568, 320), (320, 1), 0), reinterpret_tensor(arg248_1, (320, 1280), (1, 320), 0), out=buf393)
        del arg248_1
        buf394 = reinterpret_tensor(buf393, (8, 196, 1280), (250880, 1280, 1), 0); del buf393  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf394, arg249_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg249_1
        buf395 = reinterpret_tensor(buf392, (1568, 320), (320, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg250_1, (1280, 320), (1, 1280), 0), out=buf395)
        del arg250_1
        buf399 = reinterpret_tensor(buf366, (8, 196, 320), (62720, 320, 1), 0); del buf366  # reuse
        # Source Nodes: [l__mod___blocks_2_6_norm1, x_233], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf388, buf395, arg251_1, arg252_1, arg253_1, buf399, 1568, 320, grid=grid(1568), stream=stream0)
        del arg252_1
        del arg253_1
        buf400 = buf371; del buf371  # reuse
        # Source Nodes: [l__mod___blocks_2_6_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg256_1, buf400, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg256_1
        # Source Nodes: [l__mod___blocks_2_6_attn_sr], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(reinterpret_tensor(buf399, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf400, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf402 = buf375; del buf375  # reuse
        buf403 = buf374; del buf374  # reuse
        buf404 = buf373; del buf373  # reuse
        # Source Nodes: [x_236], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf401, arg257_1, buf402, buf403, buf404, 1176, 107, grid=grid(1176), stream=stream0)
        buf405 = buf377; del buf377  # reuse
        buf406 = buf376; del buf376  # reuse
        # Source Nodes: [x_236], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf402, buf403, buf404, buf405, buf406, 392, 3, grid=grid(392), stream=stream0)
        buf408 = buf379; del buf379  # reuse
        # Source Nodes: [x_236], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf401, arg257_1, buf405, buf406, arg258_1, arg259_1, buf408, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del buf401
        buf409 = buf380; del buf380  # reuse
        # Source Nodes: [l__mod___blocks_2_6_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg261_1, reinterpret_tensor(buf408, (392, 320), (320, 1), 0), reinterpret_tensor(arg260_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf409)
        del arg260_1
        del arg261_1
        buf410 = reinterpret_tensor(buf359, (1568, 320), (320, 1), 0); del buf359  # reuse
        # Source Nodes: [l__mod___blocks_2_6_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg255_1, reinterpret_tensor(buf399, (1568, 320), (320, 1), 0), reinterpret_tensor(arg254_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf410)
        del arg254_1
        del arg255_1
        del buf399
        # Source Nodes: [x_237], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf411 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf410, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf409, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf409, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf412 = buf411[0]
        del buf411
        buf416 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (1568, 320), (320, 1), 0), reinterpret_tensor(arg262_1, (320, 320), (1, 320), 0), out=buf416)
        del arg262_1
        buf417 = reinterpret_tensor(buf416, (8, 196, 320), (62720, 320, 1), 0); del buf416  # reuse
        buf421 = reinterpret_tensor(buf412, (8, 196, 320), (62720, 320, 1), 0); del buf412  # reuse
        # Source Nodes: [l__mod___blocks_2_6_norm2, x_233, x_241], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf417, buf388, buf395, arg251_1, arg263_1, arg264_1, arg265_1, buf421, 1568, 320, grid=grid(1568), stream=stream0)
        del arg251_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf422 = reinterpret_tensor(buf394, (1568, 1280), (1280, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (1568, 320), (320, 1), 0), reinterpret_tensor(arg266_1, (320, 1280), (1, 320), 0), out=buf422)
        del arg266_1
        buf423 = reinterpret_tensor(buf422, (8, 196, 1280), (250880, 1280, 1), 0); del buf422  # reuse
        # Source Nodes: [x_243], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf423, arg267_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg267_1
        buf424 = reinterpret_tensor(buf421, (1568, 320), (320, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg268_1, (1280, 320), (1, 1280), 0), out=buf424)
        del arg268_1
        buf428 = reinterpret_tensor(buf395, (8, 196, 320), (62720, 320, 1), 0); del buf395  # reuse
        # Source Nodes: [l__mod___blocks_2_7_norm1, x_249], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf417, buf424, arg269_1, arg270_1, arg271_1, buf428, 1568, 320, grid=grid(1568), stream=stream0)
        del arg270_1
        del arg271_1
        buf429 = buf400; del buf400  # reuse
        # Source Nodes: [l__mod___blocks_2_7_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg274_1, buf429, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg274_1
        # Source Nodes: [l__mod___blocks_2_7_attn_sr], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(reinterpret_tensor(buf428, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf429, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf431 = buf404; del buf404  # reuse
        buf432 = buf403; del buf403  # reuse
        buf433 = buf402; del buf402  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf430, arg275_1, buf431, buf432, buf433, 1176, 107, grid=grid(1176), stream=stream0)
        buf434 = buf406; del buf406  # reuse
        buf435 = buf405; del buf405  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf431, buf432, buf433, buf434, buf435, 392, 3, grid=grid(392), stream=stream0)
        buf437 = buf408; del buf408  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf430, arg275_1, buf434, buf435, arg276_1, arg277_1, buf437, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg275_1
        del arg276_1
        del arg277_1
        del buf430
        buf438 = buf409; del buf409  # reuse
        # Source Nodes: [l__mod___blocks_2_7_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg279_1, reinterpret_tensor(buf437, (392, 320), (320, 1), 0), reinterpret_tensor(arg278_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf438)
        del arg278_1
        del arg279_1
        buf439 = reinterpret_tensor(buf388, (1568, 320), (320, 1), 0); del buf388  # reuse
        # Source Nodes: [l__mod___blocks_2_7_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg273_1, reinterpret_tensor(buf428, (1568, 320), (320, 1), 0), reinterpret_tensor(arg272_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf439)
        del arg272_1
        del arg273_1
        del buf428
        # Source Nodes: [x_253], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf440 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf439, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf438, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf438, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf441 = buf440[0]
        del buf440
        buf445 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (1568, 320), (320, 1), 0), reinterpret_tensor(arg280_1, (320, 320), (1, 320), 0), out=buf445)
        del arg280_1
        buf446 = reinterpret_tensor(buf445, (8, 196, 320), (62720, 320, 1), 0); del buf445  # reuse
        buf450 = reinterpret_tensor(buf441, (8, 196, 320), (62720, 320, 1), 0); del buf441  # reuse
        # Source Nodes: [l__mod___blocks_2_7_norm2, x_249, x_257], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf446, buf417, buf424, arg269_1, arg281_1, arg282_1, arg283_1, buf450, 1568, 320, grid=grid(1568), stream=stream0)
        del arg269_1
        del arg281_1
        del arg282_1
        del arg283_1
        buf451 = reinterpret_tensor(buf423, (1568, 1280), (1280, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 320), (320, 1), 0), reinterpret_tensor(arg284_1, (320, 1280), (1, 320), 0), out=buf451)
        del arg284_1
        buf452 = reinterpret_tensor(buf451, (8, 196, 1280), (250880, 1280, 1), 0); del buf451  # reuse
        # Source Nodes: [x_259], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf452, arg285_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg285_1
        buf453 = reinterpret_tensor(buf450, (1568, 320), (320, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg286_1, (1280, 320), (1, 1280), 0), out=buf453)
        del arg286_1
        buf457 = reinterpret_tensor(buf424, (8, 196, 320), (62720, 320, 1), 0); del buf424  # reuse
        # Source Nodes: [l__mod___blocks_2_8_norm1, x_265], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf446, buf453, arg287_1, arg288_1, arg289_1, buf457, 1568, 320, grid=grid(1568), stream=stream0)
        del arg288_1
        del arg289_1
        buf458 = buf429; del buf429  # reuse
        # Source Nodes: [l__mod___blocks_2_8_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg292_1, buf458, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg292_1
        # Source Nodes: [l__mod___blocks_2_8_attn_sr], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(reinterpret_tensor(buf457, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf458, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf460 = buf433; del buf433  # reuse
        buf461 = buf432; del buf432  # reuse
        buf462 = buf431; del buf431  # reuse
        # Source Nodes: [x_268], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf459, arg293_1, buf460, buf461, buf462, 1176, 107, grid=grid(1176), stream=stream0)
        buf463 = buf435; del buf435  # reuse
        buf464 = buf434; del buf434  # reuse
        # Source Nodes: [x_268], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf460, buf461, buf462, buf463, buf464, 392, 3, grid=grid(392), stream=stream0)
        buf466 = buf437; del buf437  # reuse
        # Source Nodes: [x_268], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf459, arg293_1, buf463, buf464, arg294_1, arg295_1, buf466, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg293_1
        del arg294_1
        del arg295_1
        del buf459
        buf467 = buf438; del buf438  # reuse
        # Source Nodes: [l__mod___blocks_2_8_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg297_1, reinterpret_tensor(buf466, (392, 320), (320, 1), 0), reinterpret_tensor(arg296_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf467)
        del arg296_1
        del arg297_1
        buf468 = reinterpret_tensor(buf417, (1568, 320), (320, 1), 0); del buf417  # reuse
        # Source Nodes: [l__mod___blocks_2_8_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg291_1, reinterpret_tensor(buf457, (1568, 320), (320, 1), 0), reinterpret_tensor(arg290_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf468)
        del arg290_1
        del arg291_1
        del buf457
        # Source Nodes: [x_269], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf469 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf468, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf467, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf467, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf470 = buf469[0]
        del buf469
        buf474 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf470, (1568, 320), (320, 1), 0), reinterpret_tensor(arg298_1, (320, 320), (1, 320), 0), out=buf474)
        del arg298_1
        buf475 = reinterpret_tensor(buf474, (8, 196, 320), (62720, 320, 1), 0); del buf474  # reuse
        buf479 = reinterpret_tensor(buf470, (8, 196, 320), (62720, 320, 1), 0); del buf470  # reuse
        # Source Nodes: [l__mod___blocks_2_8_norm2, x_265, x_273], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf475, buf446, buf453, arg287_1, arg299_1, arg300_1, arg301_1, buf479, 1568, 320, grid=grid(1568), stream=stream0)
        del arg287_1
        del arg299_1
        del arg300_1
        del arg301_1
        buf480 = reinterpret_tensor(buf452, (1568, 1280), (1280, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (1568, 320), (320, 1), 0), reinterpret_tensor(arg302_1, (320, 1280), (1, 320), 0), out=buf480)
        del arg302_1
        buf481 = reinterpret_tensor(buf480, (8, 196, 1280), (250880, 1280, 1), 0); del buf480  # reuse
        # Source Nodes: [x_275], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf481, arg303_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg303_1
        buf482 = reinterpret_tensor(buf479, (1568, 320), (320, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg304_1, (1280, 320), (1, 1280), 0), out=buf482)
        del arg304_1
        buf486 = reinterpret_tensor(buf453, (8, 196, 320), (62720, 320, 1), 0); del buf453  # reuse
        # Source Nodes: [l__mod___blocks_2_9_norm1, x_281], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf475, buf482, arg305_1, arg306_1, arg307_1, buf486, 1568, 320, grid=grid(1568), stream=stream0)
        del arg306_1
        del arg307_1
        buf487 = buf458; del buf458  # reuse
        # Source Nodes: [l__mod___blocks_2_9_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg310_1, buf487, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg310_1
        # Source Nodes: [l__mod___blocks_2_9_attn_sr], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(reinterpret_tensor(buf486, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf487, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf489 = buf462; del buf462  # reuse
        buf490 = buf461; del buf461  # reuse
        buf491 = buf460; del buf460  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf488, arg311_1, buf489, buf490, buf491, 1176, 107, grid=grid(1176), stream=stream0)
        buf492 = buf464; del buf464  # reuse
        buf493 = buf463; del buf463  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf489, buf490, buf491, buf492, buf493, 392, 3, grid=grid(392), stream=stream0)
        buf495 = buf466; del buf466  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf488, arg311_1, buf492, buf493, arg312_1, arg313_1, buf495, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg311_1
        del arg312_1
        del arg313_1
        del buf488
        buf496 = buf467; del buf467  # reuse
        # Source Nodes: [l__mod___blocks_2_9_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg315_1, reinterpret_tensor(buf495, (392, 320), (320, 1), 0), reinterpret_tensor(arg314_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf496)
        del arg314_1
        del arg315_1
        buf497 = reinterpret_tensor(buf446, (1568, 320), (320, 1), 0); del buf446  # reuse
        # Source Nodes: [l__mod___blocks_2_9_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg309_1, reinterpret_tensor(buf486, (1568, 320), (320, 1), 0), reinterpret_tensor(arg308_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf497)
        del arg308_1
        del arg309_1
        del buf486
        # Source Nodes: [x_285], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf498 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf497, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf496, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf496, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf499 = buf498[0]
        del buf498
        buf503 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf499, (1568, 320), (320, 1), 0), reinterpret_tensor(arg316_1, (320, 320), (1, 320), 0), out=buf503)
        del arg316_1
        buf504 = reinterpret_tensor(buf503, (8, 196, 320), (62720, 320, 1), 0); del buf503  # reuse
        buf508 = reinterpret_tensor(buf499, (8, 196, 320), (62720, 320, 1), 0); del buf499  # reuse
        # Source Nodes: [l__mod___blocks_2_9_norm2, x_281, x_289], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf504, buf475, buf482, arg305_1, arg317_1, arg318_1, arg319_1, buf508, 1568, 320, grid=grid(1568), stream=stream0)
        del arg305_1
        del arg317_1
        del arg318_1
        del arg319_1
        buf509 = reinterpret_tensor(buf481, (1568, 1280), (1280, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (1568, 320), (320, 1), 0), reinterpret_tensor(arg320_1, (320, 1280), (1, 320), 0), out=buf509)
        del arg320_1
        buf510 = reinterpret_tensor(buf509, (8, 196, 1280), (250880, 1280, 1), 0); del buf509  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf510, arg321_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg321_1
        buf511 = reinterpret_tensor(buf508, (1568, 320), (320, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf510, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg322_1, (1280, 320), (1, 1280), 0), out=buf511)
        del arg322_1
        buf515 = reinterpret_tensor(buf482, (8, 196, 320), (62720, 320, 1), 0); del buf482  # reuse
        # Source Nodes: [l__mod___blocks_2_10_norm1, x_297], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf504, buf511, arg323_1, arg324_1, arg325_1, buf515, 1568, 320, grid=grid(1568), stream=stream0)
        del arg324_1
        del arg325_1
        buf516 = buf487; del buf487  # reuse
        # Source Nodes: [l__mod___blocks_2_10_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg328_1, buf516, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg328_1
        # Source Nodes: [l__mod___blocks_2_10_attn_sr], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(reinterpret_tensor(buf515, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf516, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf518 = buf491; del buf491  # reuse
        buf519 = buf490; del buf490  # reuse
        buf520 = buf489; del buf489  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf517, arg329_1, buf518, buf519, buf520, 1176, 107, grid=grid(1176), stream=stream0)
        buf521 = buf493; del buf493  # reuse
        buf522 = buf492; del buf492  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf518, buf519, buf520, buf521, buf522, 392, 3, grid=grid(392), stream=stream0)
        buf524 = buf495; del buf495  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf517, arg329_1, buf521, buf522, arg330_1, arg331_1, buf524, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg329_1
        del arg330_1
        del arg331_1
        del buf517
        buf525 = buf496; del buf496  # reuse
        # Source Nodes: [l__mod___blocks_2_10_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg333_1, reinterpret_tensor(buf524, (392, 320), (320, 1), 0), reinterpret_tensor(arg332_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf525)
        del arg332_1
        del arg333_1
        buf526 = reinterpret_tensor(buf475, (1568, 320), (320, 1), 0); del buf475  # reuse
        # Source Nodes: [l__mod___blocks_2_10_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg327_1, reinterpret_tensor(buf515, (1568, 320), (320, 1), 0), reinterpret_tensor(arg326_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf526)
        del arg326_1
        del arg327_1
        del buf515
        # Source Nodes: [x_301], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf527 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf526, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf525, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf528 = buf527[0]
        del buf527
        buf532 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (1568, 320), (320, 1), 0), reinterpret_tensor(arg334_1, (320, 320), (1, 320), 0), out=buf532)
        del arg334_1
        buf533 = reinterpret_tensor(buf532, (8, 196, 320), (62720, 320, 1), 0); del buf532  # reuse
        buf537 = reinterpret_tensor(buf528, (8, 196, 320), (62720, 320, 1), 0); del buf528  # reuse
        # Source Nodes: [l__mod___blocks_2_10_norm2, x_297, x_305], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf533, buf504, buf511, arg323_1, arg335_1, arg336_1, arg337_1, buf537, 1568, 320, grid=grid(1568), stream=stream0)
        del arg323_1
        del arg335_1
        del arg336_1
        del arg337_1
        buf538 = reinterpret_tensor(buf510, (1568, 1280), (1280, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf537, (1568, 320), (320, 1), 0), reinterpret_tensor(arg338_1, (320, 1280), (1, 320), 0), out=buf538)
        del arg338_1
        buf539 = reinterpret_tensor(buf538, (8, 196, 1280), (250880, 1280, 1), 0); del buf538  # reuse
        # Source Nodes: [x_307], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf539, arg339_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg339_1
        buf540 = reinterpret_tensor(buf537, (1568, 320), (320, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf539, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg340_1, (1280, 320), (1, 1280), 0), out=buf540)
        del arg340_1
        buf544 = reinterpret_tensor(buf511, (8, 196, 320), (62720, 320, 1), 0); del buf511  # reuse
        # Source Nodes: [l__mod___blocks_2_11_norm1, x_313], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf533, buf540, arg341_1, arg342_1, arg343_1, buf544, 1568, 320, grid=grid(1568), stream=stream0)
        del arg342_1
        del arg343_1
        buf545 = buf516; del buf516  # reuse
        # Source Nodes: [l__mod___blocks_2_11_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg346_1, buf545, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg346_1
        # Source Nodes: [l__mod___blocks_2_11_attn_sr], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(reinterpret_tensor(buf544, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf545, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf547 = buf520; del buf520  # reuse
        buf548 = buf519; del buf519  # reuse
        buf549 = buf518; del buf518  # reuse
        # Source Nodes: [x_316], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf546, arg347_1, buf547, buf548, buf549, 1176, 107, grid=grid(1176), stream=stream0)
        buf550 = buf522; del buf522  # reuse
        buf551 = buf521; del buf521  # reuse
        # Source Nodes: [x_316], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf547, buf548, buf549, buf550, buf551, 392, 3, grid=grid(392), stream=stream0)
        buf553 = buf524; del buf524  # reuse
        # Source Nodes: [x_316], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf546, arg347_1, buf550, buf551, arg348_1, arg349_1, buf553, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del buf546
        buf554 = buf525; del buf525  # reuse
        # Source Nodes: [l__mod___blocks_2_11_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg351_1, reinterpret_tensor(buf553, (392, 320), (320, 1), 0), reinterpret_tensor(arg350_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf554)
        del arg350_1
        del arg351_1
        buf555 = reinterpret_tensor(buf504, (1568, 320), (320, 1), 0); del buf504  # reuse
        # Source Nodes: [l__mod___blocks_2_11_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg345_1, reinterpret_tensor(buf544, (1568, 320), (320, 1), 0), reinterpret_tensor(arg344_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf555)
        del arg344_1
        del arg345_1
        del buf544
        # Source Nodes: [x_317], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf556 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf555, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf554, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf554, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf557 = buf556[0]
        del buf556
        buf561 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 320), (320, 1), 0), reinterpret_tensor(arg352_1, (320, 320), (1, 320), 0), out=buf561)
        del arg352_1
        buf562 = reinterpret_tensor(buf561, (8, 196, 320), (62720, 320, 1), 0); del buf561  # reuse
        buf566 = reinterpret_tensor(buf557, (8, 196, 320), (62720, 320, 1), 0); del buf557  # reuse
        # Source Nodes: [l__mod___blocks_2_11_norm2, x_313, x_321], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf562, buf533, buf540, arg341_1, arg353_1, arg354_1, arg355_1, buf566, 1568, 320, grid=grid(1568), stream=stream0)
        del arg341_1
        del arg353_1
        del arg354_1
        del arg355_1
        buf567 = reinterpret_tensor(buf539, (1568, 1280), (1280, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf566, (1568, 320), (320, 1), 0), reinterpret_tensor(arg356_1, (320, 1280), (1, 320), 0), out=buf567)
        del arg356_1
        buf568 = reinterpret_tensor(buf567, (8, 196, 1280), (250880, 1280, 1), 0); del buf567  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf568, arg357_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg357_1
        buf569 = reinterpret_tensor(buf566, (1568, 320), (320, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg358_1, (1280, 320), (1, 1280), 0), out=buf569)
        del arg358_1
        buf573 = reinterpret_tensor(buf540, (8, 196, 320), (62720, 320, 1), 0); del buf540  # reuse
        # Source Nodes: [l__mod___blocks_2_12_norm1, x_329], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf562, buf569, arg359_1, arg360_1, arg361_1, buf573, 1568, 320, grid=grid(1568), stream=stream0)
        del arg360_1
        del arg361_1
        buf574 = buf545; del buf545  # reuse
        # Source Nodes: [l__mod___blocks_2_12_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg364_1, buf574, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg364_1
        # Source Nodes: [l__mod___blocks_2_12_attn_sr], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(reinterpret_tensor(buf573, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf574, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf576 = buf549; del buf549  # reuse
        buf577 = buf548; del buf548  # reuse
        buf578 = buf547; del buf547  # reuse
        # Source Nodes: [x_332], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf575, arg365_1, buf576, buf577, buf578, 1176, 107, grid=grid(1176), stream=stream0)
        buf579 = buf551; del buf551  # reuse
        buf580 = buf550; del buf550  # reuse
        # Source Nodes: [x_332], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf576, buf577, buf578, buf579, buf580, 392, 3, grid=grid(392), stream=stream0)
        buf582 = buf553; del buf553  # reuse
        # Source Nodes: [x_332], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf575, arg365_1, buf579, buf580, arg366_1, arg367_1, buf582, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg365_1
        del arg366_1
        del arg367_1
        del buf575
        buf583 = buf554; del buf554  # reuse
        # Source Nodes: [l__mod___blocks_2_12_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg369_1, reinterpret_tensor(buf582, (392, 320), (320, 1), 0), reinterpret_tensor(arg368_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf583)
        del arg368_1
        del arg369_1
        buf584 = reinterpret_tensor(buf533, (1568, 320), (320, 1), 0); del buf533  # reuse
        # Source Nodes: [l__mod___blocks_2_12_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg363_1, reinterpret_tensor(buf573, (1568, 320), (320, 1), 0), reinterpret_tensor(arg362_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf584)
        del arg362_1
        del arg363_1
        del buf573
        # Source Nodes: [x_333], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf585 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf584, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf583, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf583, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf586 = buf585[0]
        del buf585
        buf590 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf586, (1568, 320), (320, 1), 0), reinterpret_tensor(arg370_1, (320, 320), (1, 320), 0), out=buf590)
        del arg370_1
        buf591 = reinterpret_tensor(buf590, (8, 196, 320), (62720, 320, 1), 0); del buf590  # reuse
        buf595 = reinterpret_tensor(buf586, (8, 196, 320), (62720, 320, 1), 0); del buf586  # reuse
        # Source Nodes: [l__mod___blocks_2_12_norm2, x_329, x_337], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf591, buf562, buf569, arg359_1, arg371_1, arg372_1, arg373_1, buf595, 1568, 320, grid=grid(1568), stream=stream0)
        del arg359_1
        del arg371_1
        del arg372_1
        del arg373_1
        buf596 = reinterpret_tensor(buf568, (1568, 1280), (1280, 1), 0); del buf568  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf595, (1568, 320), (320, 1), 0), reinterpret_tensor(arg374_1, (320, 1280), (1, 320), 0), out=buf596)
        del arg374_1
        buf597 = reinterpret_tensor(buf596, (8, 196, 1280), (250880, 1280, 1), 0); del buf596  # reuse
        # Source Nodes: [x_339], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf597, arg375_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg375_1
        buf598 = reinterpret_tensor(buf595, (1568, 320), (320, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf597, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg376_1, (1280, 320), (1, 1280), 0), out=buf598)
        del arg376_1
        buf602 = reinterpret_tensor(buf569, (8, 196, 320), (62720, 320, 1), 0); del buf569  # reuse
        # Source Nodes: [l__mod___blocks_2_13_norm1, x_345], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf591, buf598, arg377_1, arg378_1, arg379_1, buf602, 1568, 320, grid=grid(1568), stream=stream0)
        del arg378_1
        del arg379_1
        buf603 = buf574; del buf574  # reuse
        # Source Nodes: [l__mod___blocks_2_13_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg382_1, buf603, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg382_1
        # Source Nodes: [l__mod___blocks_2_13_attn_sr], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(reinterpret_tensor(buf602, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf603, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf604, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf605 = buf578; del buf578  # reuse
        buf606 = buf577; del buf577  # reuse
        buf607 = buf576; del buf576  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf604, arg383_1, buf605, buf606, buf607, 1176, 107, grid=grid(1176), stream=stream0)
        buf608 = buf580; del buf580  # reuse
        buf609 = buf579; del buf579  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf605, buf606, buf607, buf608, buf609, 392, 3, grid=grid(392), stream=stream0)
        buf611 = buf582; del buf582  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf604, arg383_1, buf608, buf609, arg384_1, arg385_1, buf611, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg383_1
        del arg384_1
        del arg385_1
        del buf604
        buf612 = buf583; del buf583  # reuse
        # Source Nodes: [l__mod___blocks_2_13_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg387_1, reinterpret_tensor(buf611, (392, 320), (320, 1), 0), reinterpret_tensor(arg386_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf612)
        del arg386_1
        del arg387_1
        buf613 = reinterpret_tensor(buf562, (1568, 320), (320, 1), 0); del buf562  # reuse
        # Source Nodes: [l__mod___blocks_2_13_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg381_1, reinterpret_tensor(buf602, (1568, 320), (320, 1), 0), reinterpret_tensor(arg380_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf613)
        del arg380_1
        del arg381_1
        del buf602
        # Source Nodes: [x_349], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf614 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf613, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf612, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf612, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf615 = buf614[0]
        del buf614
        buf619 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf615, (1568, 320), (320, 1), 0), reinterpret_tensor(arg388_1, (320, 320), (1, 320), 0), out=buf619)
        del arg388_1
        buf620 = reinterpret_tensor(buf619, (8, 196, 320), (62720, 320, 1), 0); del buf619  # reuse
        buf624 = reinterpret_tensor(buf615, (8, 196, 320), (62720, 320, 1), 0); del buf615  # reuse
        # Source Nodes: [l__mod___blocks_2_13_norm2, x_345, x_353], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf620, buf591, buf598, arg377_1, arg389_1, arg390_1, arg391_1, buf624, 1568, 320, grid=grid(1568), stream=stream0)
        del arg377_1
        del arg389_1
        del arg390_1
        del arg391_1
        buf625 = reinterpret_tensor(buf597, (1568, 1280), (1280, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf624, (1568, 320), (320, 1), 0), reinterpret_tensor(arg392_1, (320, 1280), (1, 320), 0), out=buf625)
        del arg392_1
        buf626 = reinterpret_tensor(buf625, (8, 196, 1280), (250880, 1280, 1), 0); del buf625  # reuse
        # Source Nodes: [x_355], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf626, arg393_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg393_1
        buf627 = reinterpret_tensor(buf624, (1568, 320), (320, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf626, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg394_1, (1280, 320), (1, 1280), 0), out=buf627)
        del arg394_1
        buf631 = reinterpret_tensor(buf598, (8, 196, 320), (62720, 320, 1), 0); del buf598  # reuse
        # Source Nodes: [l__mod___blocks_2_14_norm1, x_361], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf620, buf627, arg395_1, arg396_1, arg397_1, buf631, 1568, 320, grid=grid(1568), stream=stream0)
        del arg396_1
        del arg397_1
        buf632 = buf603; del buf603  # reuse
        # Source Nodes: [l__mod___blocks_2_14_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg400_1, buf632, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg400_1
        # Source Nodes: [l__mod___blocks_2_14_attn_sr], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(reinterpret_tensor(buf631, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf632, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf634 = buf607; del buf607  # reuse
        buf635 = buf606; del buf606  # reuse
        buf636 = buf605; del buf605  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf633, arg401_1, buf634, buf635, buf636, 1176, 107, grid=grid(1176), stream=stream0)
        buf637 = buf609; del buf609  # reuse
        buf638 = buf608; del buf608  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf634, buf635, buf636, buf637, buf638, 392, 3, grid=grid(392), stream=stream0)
        buf640 = buf611; del buf611  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf633, arg401_1, buf637, buf638, arg402_1, arg403_1, buf640, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg401_1
        del arg402_1
        del arg403_1
        del buf633
        buf641 = buf612; del buf612  # reuse
        # Source Nodes: [l__mod___blocks_2_14_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg405_1, reinterpret_tensor(buf640, (392, 320), (320, 1), 0), reinterpret_tensor(arg404_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf641)
        del arg404_1
        del arg405_1
        buf642 = reinterpret_tensor(buf591, (1568, 320), (320, 1), 0); del buf591  # reuse
        # Source Nodes: [l__mod___blocks_2_14_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg399_1, reinterpret_tensor(buf631, (1568, 320), (320, 1), 0), reinterpret_tensor(arg398_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf642)
        del arg398_1
        del arg399_1
        del buf631
        # Source Nodes: [x_365], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf643 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf642, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf641, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf641, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf644 = buf643[0]
        del buf643
        buf648 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (1568, 320), (320, 1), 0), reinterpret_tensor(arg406_1, (320, 320), (1, 320), 0), out=buf648)
        del arg406_1
        buf649 = reinterpret_tensor(buf648, (8, 196, 320), (62720, 320, 1), 0); del buf648  # reuse
        buf653 = reinterpret_tensor(buf644, (8, 196, 320), (62720, 320, 1), 0); del buf644  # reuse
        # Source Nodes: [l__mod___blocks_2_14_norm2, x_361, x_369], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf649, buf620, buf627, arg395_1, arg407_1, arg408_1, arg409_1, buf653, 1568, 320, grid=grid(1568), stream=stream0)
        del arg395_1
        del arg407_1
        del arg408_1
        del arg409_1
        buf654 = reinterpret_tensor(buf626, (1568, 1280), (1280, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (1568, 320), (320, 1), 0), reinterpret_tensor(arg410_1, (320, 1280), (1, 320), 0), out=buf654)
        del arg410_1
        buf655 = reinterpret_tensor(buf654, (8, 196, 1280), (250880, 1280, 1), 0); del buf654  # reuse
        # Source Nodes: [x_371], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf655, arg411_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg411_1
        buf656 = reinterpret_tensor(buf653, (1568, 320), (320, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf655, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg412_1, (1280, 320), (1, 1280), 0), out=buf656)
        del arg412_1
        buf660 = reinterpret_tensor(buf627, (8, 196, 320), (62720, 320, 1), 0); del buf627  # reuse
        # Source Nodes: [l__mod___blocks_2_15_norm1, x_377], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf649, buf656, arg413_1, arg414_1, arg415_1, buf660, 1568, 320, grid=grid(1568), stream=stream0)
        del arg414_1
        del arg415_1
        buf661 = buf632; del buf632  # reuse
        # Source Nodes: [l__mod___blocks_2_15_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg418_1, buf661, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg418_1
        # Source Nodes: [l__mod___blocks_2_15_attn_sr], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(reinterpret_tensor(buf660, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf661, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf663 = buf636; del buf636  # reuse
        buf664 = buf635; del buf635  # reuse
        buf665 = buf634; del buf634  # reuse
        # Source Nodes: [x_380], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf662, arg419_1, buf663, buf664, buf665, 1176, 107, grid=grid(1176), stream=stream0)
        buf666 = buf638; del buf638  # reuse
        buf667 = buf637; del buf637  # reuse
        # Source Nodes: [x_380], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf663, buf664, buf665, buf666, buf667, 392, 3, grid=grid(392), stream=stream0)
        buf669 = buf640; del buf640  # reuse
        # Source Nodes: [x_380], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf662, arg419_1, buf666, buf667, arg420_1, arg421_1, buf669, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg419_1
        del arg420_1
        del arg421_1
        del buf662
        buf670 = buf641; del buf641  # reuse
        # Source Nodes: [l__mod___blocks_2_15_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg423_1, reinterpret_tensor(buf669, (392, 320), (320, 1), 0), reinterpret_tensor(arg422_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf670)
        del arg422_1
        del arg423_1
        buf671 = reinterpret_tensor(buf620, (1568, 320), (320, 1), 0); del buf620  # reuse
        # Source Nodes: [l__mod___blocks_2_15_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg417_1, reinterpret_tensor(buf660, (1568, 320), (320, 1), 0), reinterpret_tensor(arg416_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf671)
        del arg416_1
        del arg417_1
        del buf660
        # Source Nodes: [x_381], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf672 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf671, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf670, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf670, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf673 = buf672[0]
        del buf672
        buf677 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf673, (1568, 320), (320, 1), 0), reinterpret_tensor(arg424_1, (320, 320), (1, 320), 0), out=buf677)
        del arg424_1
        buf678 = reinterpret_tensor(buf677, (8, 196, 320), (62720, 320, 1), 0); del buf677  # reuse
        buf682 = reinterpret_tensor(buf673, (8, 196, 320), (62720, 320, 1), 0); del buf673  # reuse
        # Source Nodes: [l__mod___blocks_2_15_norm2, x_377, x_385], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf678, buf649, buf656, arg413_1, arg425_1, arg426_1, arg427_1, buf682, 1568, 320, grid=grid(1568), stream=stream0)
        del arg413_1
        del arg425_1
        del arg426_1
        del arg427_1
        buf683 = reinterpret_tensor(buf655, (1568, 1280), (1280, 1), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (1568, 320), (320, 1), 0), reinterpret_tensor(arg428_1, (320, 1280), (1, 320), 0), out=buf683)
        del arg428_1
        buf684 = reinterpret_tensor(buf683, (8, 196, 1280), (250880, 1280, 1), 0); del buf683  # reuse
        # Source Nodes: [x_387], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf684, arg429_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg429_1
        buf685 = reinterpret_tensor(buf682, (1568, 320), (320, 1), 0); del buf682  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg430_1, (1280, 320), (1, 1280), 0), out=buf685)
        del arg430_1
        buf689 = reinterpret_tensor(buf656, (8, 196, 320), (62720, 320, 1), 0); del buf656  # reuse
        # Source Nodes: [l__mod___blocks_2_16_norm1, x_393], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf678, buf685, arg431_1, arg432_1, arg433_1, buf689, 1568, 320, grid=grid(1568), stream=stream0)
        del arg432_1
        del arg433_1
        buf690 = buf661; del buf661  # reuse
        # Source Nodes: [l__mod___blocks_2_16_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg436_1, buf690, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg436_1
        # Source Nodes: [l__mod___blocks_2_16_attn_sr], Original ATen: [aten.convolution]
        buf691 = extern_kernels.convolution(reinterpret_tensor(buf689, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf690, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf692 = buf665; del buf665  # reuse
        buf693 = buf664; del buf664  # reuse
        buf694 = buf663; del buf663  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf691, arg437_1, buf692, buf693, buf694, 1176, 107, grid=grid(1176), stream=stream0)
        buf695 = buf667; del buf667  # reuse
        buf696 = buf666; del buf666  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf692, buf693, buf694, buf695, buf696, 392, 3, grid=grid(392), stream=stream0)
        buf698 = buf669; del buf669  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf691, arg437_1, buf695, buf696, arg438_1, arg439_1, buf698, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del buf691
        buf699 = buf670; del buf670  # reuse
        # Source Nodes: [l__mod___blocks_2_16_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg441_1, reinterpret_tensor(buf698, (392, 320), (320, 1), 0), reinterpret_tensor(arg440_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf699)
        del arg440_1
        del arg441_1
        buf700 = reinterpret_tensor(buf649, (1568, 320), (320, 1), 0); del buf649  # reuse
        # Source Nodes: [l__mod___blocks_2_16_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg435_1, reinterpret_tensor(buf689, (1568, 320), (320, 1), 0), reinterpret_tensor(arg434_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf700)
        del arg434_1
        del arg435_1
        del buf689
        # Source Nodes: [x_397], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf701 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf700, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf699, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf699, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf702 = buf701[0]
        del buf701
        buf706 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf702, (1568, 320), (320, 1), 0), reinterpret_tensor(arg442_1, (320, 320), (1, 320), 0), out=buf706)
        del arg442_1
        buf707 = reinterpret_tensor(buf706, (8, 196, 320), (62720, 320, 1), 0); del buf706  # reuse
        buf711 = reinterpret_tensor(buf702, (8, 196, 320), (62720, 320, 1), 0); del buf702  # reuse
        # Source Nodes: [l__mod___blocks_2_16_norm2, x_393, x_401], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf707, buf678, buf685, arg431_1, arg443_1, arg444_1, arg445_1, buf711, 1568, 320, grid=grid(1568), stream=stream0)
        del arg431_1
        del arg443_1
        del arg444_1
        del arg445_1
        buf712 = reinterpret_tensor(buf684, (1568, 1280), (1280, 1), 0); del buf684  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf711, (1568, 320), (320, 1), 0), reinterpret_tensor(arg446_1, (320, 1280), (1, 320), 0), out=buf712)
        del arg446_1
        buf713 = reinterpret_tensor(buf712, (8, 196, 1280), (250880, 1280, 1), 0); del buf712  # reuse
        # Source Nodes: [x_403], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf713, arg447_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg447_1
        buf714 = reinterpret_tensor(buf711, (1568, 320), (320, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf713, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg448_1, (1280, 320), (1, 1280), 0), out=buf714)
        del arg448_1
        buf718 = reinterpret_tensor(buf685, (8, 196, 320), (62720, 320, 1), 0); del buf685  # reuse
        # Source Nodes: [l__mod___blocks_2_17_norm1, x_409], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf707, buf714, arg449_1, arg450_1, arg451_1, buf718, 1568, 320, grid=grid(1568), stream=stream0)
        del arg450_1
        del arg451_1
        buf719 = buf690; del buf690  # reuse
        # Source Nodes: [l__mod___blocks_2_17_attn_sr], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg454_1, buf719, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg454_1
        # Source Nodes: [l__mod___blocks_2_17_attn_sr], Original ATen: [aten.convolution]
        buf720 = extern_kernels.convolution(reinterpret_tensor(buf718, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf719, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf720, (8, 320, 7, 7), (15680, 49, 7, 1))
        del buf719
        buf721 = buf694; del buf694  # reuse
        buf722 = buf693; del buf693  # reuse
        buf723 = buf692; del buf692  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_39.run(buf720, arg455_1, buf721, buf722, buf723, 1176, 107, grid=grid(1176), stream=stream0)
        buf724 = buf696; del buf696  # reuse
        buf725 = buf695; del buf695  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_40.run(buf721, buf722, buf723, buf724, buf725, 392, 3, grid=grid(392), stream=stream0)
        del buf721
        del buf722
        del buf723
        buf727 = buf698; del buf698  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_41.run(buf720, arg455_1, buf724, buf725, arg456_1, arg457_1, buf727, 392, 320, grid=grid(392, 320), stream=stream0)
        del arg455_1
        del arg456_1
        del arg457_1
        del buf720
        buf728 = buf699; del buf699  # reuse
        # Source Nodes: [l__mod___blocks_2_17_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg459_1, reinterpret_tensor(buf727, (392, 320), (320, 1), 0), reinterpret_tensor(arg458_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf728)
        del arg458_1
        del arg459_1
        del buf727
        buf729 = reinterpret_tensor(buf678, (1568, 320), (320, 1), 0); del buf678  # reuse
        # Source Nodes: [l__mod___blocks_2_17_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg453_1, reinterpret_tensor(buf718, (1568, 320), (320, 1), 0), reinterpret_tensor(arg452_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf729)
        del arg452_1
        del arg453_1
        del buf718
        # Source Nodes: [x_413], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf730 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf729, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf728, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf728, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        del buf728
        buf731 = buf730[0]
        del buf730
        buf735 = buf729; del buf729  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf731, (1568, 320), (320, 1), 0), reinterpret_tensor(arg460_1, (320, 320), (1, 320), 0), out=buf735)
        del arg460_1
        buf736 = reinterpret_tensor(buf735, (8, 196, 320), (62720, 320, 1), 0); del buf735  # reuse
        buf740 = reinterpret_tensor(buf731, (8, 196, 320), (62720, 320, 1), 0); del buf731  # reuse
        # Source Nodes: [l__mod___blocks_2_17_norm2, x_409, x_417], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf736, buf707, buf714, arg449_1, arg461_1, arg462_1, arg463_1, buf740, 1568, 320, grid=grid(1568), stream=stream0)
        del arg449_1
        del arg461_1
        del arg462_1
        del arg463_1
        del buf707
        del buf714
        buf741 = reinterpret_tensor(buf713, (1568, 1280), (1280, 1), 0); del buf713  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf740, (1568, 320), (320, 1), 0), reinterpret_tensor(arg464_1, (320, 1280), (1, 320), 0), out=buf741)
        del arg464_1
        buf742 = reinterpret_tensor(buf741, (8, 196, 1280), (250880, 1280, 1), 0); del buf741  # reuse
        # Source Nodes: [x_419], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf742, arg465_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg465_1
        buf743 = reinterpret_tensor(buf740, (1568, 320), (320, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg466_1, (1280, 320), (1, 1280), 0), out=buf743)
        del arg466_1
        del buf742
        buf744 = reinterpret_tensor(buf743, (8, 196, 320), (62720, 320, 1), 0); del buf743  # reuse
        # Source Nodes: [x_425], Original ATen: [aten.add]
        triton_poi_fused_add_53.run(buf744, buf736, arg467_1, 501760, grid=grid(501760), stream=stream0)
        del arg467_1
        del buf736
        buf745 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embeds_3_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(arg468_1, buf745, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del arg468_1
        # Source Nodes: [l__mod___patch_embeds_3_proj], Original ATen: [aten.convolution]
        buf746 = extern_kernels.convolution(reinterpret_tensor(buf744, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf745, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf746, (8, 512, 7, 7), (25088, 49, 7, 1))
        del buf744
        del buf745
        buf747 = reinterpret_tensor(buf252, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf252  # reuse
        buf748 = reinterpret_tensor(buf251, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf251  # reuse
        buf749 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_55.run(buf746, arg469_1, buf747, buf748, buf749, 1568, 128, grid=grid(1568), stream=stream0)
        buf750 = buf725; del buf725  # reuse
        buf751 = buf724; del buf724  # reuse
        # Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_56.run(buf747, buf748, buf749, buf750, buf751, 392, 4, grid=grid(392), stream=stream0)
        buf753 = reinterpret_tensor(buf746, (8, 49, 512), (25088, 1, 49), 0); del buf746  # reuse
        # Source Nodes: [x_429], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_57.run(buf753, arg469_1, buf750, buf751, arg470_1, arg471_1, 200704, grid=grid(200704), stream=stream0)
        del arg469_1
        del arg470_1
        del arg471_1
        buf754 = buf749; del buf749  # reuse
        buf755 = buf748; del buf748  # reuse
        buf756 = buf747; del buf747  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_58.run(buf753, buf754, buf755, buf756, 1568, 128, grid=grid(1568), stream=stream0)
        buf757 = buf751; del buf751  # reuse
        buf758 = buf750; del buf750  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_56.run(buf754, buf755, buf756, buf757, buf758, 392, 4, grid=grid(392), stream=stream0)
        buf760 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_59.run(buf753, buf757, buf758, arg472_1, arg473_1, buf760, 392, 512, grid=grid(392, 512), stream=stream0)
        del arg472_1
        del arg473_1
        buf761 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_0_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg477_1, reinterpret_tensor(buf760, (392, 512), (512, 1), 0), reinterpret_tensor(arg476_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf761)
        del arg476_1
        del arg477_1
        buf762 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg475_1, reinterpret_tensor(buf760, (392, 512), (512, 1), 0), reinterpret_tensor(arg474_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf762)
        del arg474_1
        del arg475_1
        del buf760
        # Source Nodes: [x_431], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf763 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf762, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf761, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf761, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        buf764 = buf763[0]
        del buf763
        buf768 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf764, (392, 512), (512, 1), 0), reinterpret_tensor(arg478_1, (512, 512), (1, 512), 0), out=buf768)
        del arg478_1
        buf769 = reinterpret_tensor(buf756, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf756  # reuse
        buf770 = reinterpret_tensor(buf755, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf755  # reuse
        buf771 = reinterpret_tensor(buf754, (8, 49, 1, 4), (196, 4, 1568, 1), 0); del buf754  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_60.run(buf753, buf768, arg479_1, buf769, buf770, buf771, 1568, 128, grid=grid(1568), stream=stream0)
        buf772 = buf758; del buf758  # reuse
        buf773 = buf757; del buf757  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_61.run(buf769, buf770, buf771, buf772, buf773, 392, 4, grid=grid(392), stream=stream0)
        buf775 = reinterpret_tensor(buf764, (8, 49, 512), (25088, 512, 1), 0); del buf764  # reuse
        # Source Nodes: [l__mod___blocks_3_0_norm2, x_435], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_62.run(buf753, buf768, arg479_1, buf772, buf773, arg480_1, arg481_1, buf775, 392, 512, grid=grid(392, 512), stream=stream0)
        del arg480_1
        del arg481_1
        buf776 = reinterpret_tensor(buf200, (392, 2048), (2048, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (392, 512), (512, 1), 0), reinterpret_tensor(arg482_1, (512, 2048), (1, 512), 0), out=buf776)
        del arg482_1
        buf777 = reinterpret_tensor(buf776, (8, 49, 2048), (100352, 2048, 1), 0); del buf776  # reuse
        # Source Nodes: [x_437], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_63.run(buf777, arg483_1, 802816, grid=grid(802816), stream=stream0)
        del arg483_1
        buf778 = reinterpret_tensor(buf775, (392, 512), (512, 1), 0); del buf775  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf777, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg484_1, (2048, 512), (1, 2048), 0), out=buf778)
        del arg484_1
        buf779 = reinterpret_tensor(buf768, (8, 49, 512), (25088, 512, 1), 0); del buf768  # reuse
        # Source Nodes: [x_435, x_443], Original ATen: [aten.add]
        triton_poi_fused_add_64.run(buf779, buf753, arg479_1, buf778, arg485_1, 392, 512, grid=grid(392, 512), stream=stream0)
        del arg479_1
        del arg485_1
        # Source Nodes: [x_444], Original ATen: [aten.convolution]
        buf780 = extern_kernels.convolution(reinterpret_tensor(buf779, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), arg486_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf780, (8, 512, 7, 7), (25088, 49, 7, 1))
        del arg486_1
        buf781 = buf771; del buf771  # reuse
        buf782 = buf770; del buf770  # reuse
        buf783 = buf769; del buf769  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_65.run(buf780, arg487_1, buf779, buf781, buf782, buf783, 1568, 128, grid=grid(1568), stream=stream0)
        buf784 = buf773; del buf773  # reuse
        buf785 = buf772; del buf772  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_61.run(buf781, buf782, buf783, buf784, buf785, 392, 4, grid=grid(392), stream=stream0)
        del buf781
        del buf782
        del buf783
        buf787 = reinterpret_tensor(buf778, (8, 49, 512), (25088, 512, 1), 0); del buf778  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_66.run(buf780, arg487_1, buf779, buf784, buf785, arg488_1, arg489_1, buf787, 392, 512, grid=grid(392, 512), stream=stream0)
        del arg488_1
        del arg489_1
        buf788 = buf761; del buf761  # reuse
        # Source Nodes: [l__mod___blocks_3_1_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg493_1, reinterpret_tensor(buf787, (392, 512), (512, 1), 0), reinterpret_tensor(arg492_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf788)
        del arg492_1
        del arg493_1
        buf789 = reinterpret_tensor(buf753, (392, 512), (512, 1), 0); del buf753  # reuse
        # Source Nodes: [l__mod___blocks_3_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg491_1, reinterpret_tensor(buf787, (392, 512), (512, 1), 0), reinterpret_tensor(arg490_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf789)
        del arg490_1
        del arg491_1
        del buf787
        # Source Nodes: [x_448], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf790 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf789, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf788, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf788, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        buf791 = buf790[0]
        del buf790
        buf795 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf791, (392, 512), (512, 1), 0), reinterpret_tensor(arg494_1, (512, 512), (1, 512), 0), out=buf795)
        del arg494_1
        buf796 = buf779; del buf779  # reuse
        buf800 = reinterpret_tensor(buf791, (8, 49, 512), (25088, 512, 1), 0); del buf791  # reuse
        # Source Nodes: [l__mod___blocks_3_1_norm2, x_452], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_67.run(buf796, buf780, arg487_1, buf795, arg495_1, arg496_1, arg497_1, buf800, 392, 512, grid=grid(392), stream=stream0)
        del arg487_1
        del arg495_1
        del arg496_1
        del arg497_1
        buf801 = reinterpret_tensor(buf777, (392, 2048), (2048, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf800, (392, 512), (512, 1), 0), reinterpret_tensor(arg498_1, (512, 2048), (1, 512), 0), out=buf801)
        del arg498_1
        buf802 = reinterpret_tensor(buf801, (8, 49, 2048), (100352, 2048, 1), 0); del buf801  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_63.run(buf802, arg499_1, 802816, grid=grid(802816), stream=stream0)
        del arg499_1
        buf803 = reinterpret_tensor(buf800, (392, 512), (512, 1), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf802, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg500_1, (2048, 512), (1, 2048), 0), out=buf803)
        del arg500_1
        buf807 = reinterpret_tensor(buf795, (8, 49, 512), (25088, 512, 1), 0); del buf795  # reuse
        # Source Nodes: [l__mod___blocks_3_2_norm1, x_460], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_68.run(buf796, buf803, arg501_1, arg502_1, arg503_1, buf807, 392, 512, grid=grid(392), stream=stream0)
        del arg502_1
        del arg503_1
        buf808 = buf788; del buf788  # reuse
        # Source Nodes: [l__mod___blocks_3_2_attn_kv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg507_1, reinterpret_tensor(buf807, (392, 512), (512, 1), 0), reinterpret_tensor(arg506_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf808)
        del arg506_1
        del arg507_1
        buf809 = reinterpret_tensor(buf780, (392, 512), (512, 1), 0); del buf780  # reuse
        # Source Nodes: [l__mod___blocks_3_2_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg505_1, reinterpret_tensor(buf807, (392, 512), (512, 1), 0), reinterpret_tensor(arg504_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf809)
        del arg504_1
        del arg505_1
        del buf807
        # Source Nodes: [x_461], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf810 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf809, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf808, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf808, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        del buf808
        buf811 = buf810[0]
        del buf810
        buf815 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf811, (392, 512), (512, 1), 0), reinterpret_tensor(arg508_1, (512, 512), (1, 512), 0), out=buf815)
        del arg508_1
        buf816 = reinterpret_tensor(buf815, (8, 49, 512), (25088, 512, 1), 0); del buf815  # reuse
        buf820 = reinterpret_tensor(buf811, (8, 49, 512), (25088, 512, 1), 0); del buf811  # reuse
        # Source Nodes: [l__mod___blocks_3_2_norm2, x_460, x_465], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_69.run(buf816, buf796, buf803, arg501_1, arg509_1, arg510_1, arg511_1, buf820, 392, 512, grid=grid(392), stream=stream0)
        del arg501_1
        del arg509_1
        del arg510_1
        del arg511_1
        del buf796
        del buf803
        buf821 = reinterpret_tensor(buf802, (392, 2048), (2048, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf820, (392, 512), (512, 1), 0), reinterpret_tensor(arg512_1, (512, 2048), (1, 512), 0), out=buf821)
        del arg512_1
        buf822 = reinterpret_tensor(buf821, (8, 49, 2048), (100352, 2048, 1), 0); del buf821  # reuse
        # Source Nodes: [x_467], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_63.run(buf822, arg513_1, 802816, grid=grid(802816), stream=stream0)
        del arg513_1
        buf823 = reinterpret_tensor(buf820, (392, 512), (512, 1), 0); del buf820  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf822, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg514_1, (2048, 512), (1, 2048), 0), out=buf823)
        del arg514_1
        del buf822
        buf824 = buf785; del buf785  # reuse
        buf825 = buf784; del buf784  # reuse
        # Source Nodes: [x_473, x_475], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_70.run(buf816, buf823, arg515_1, buf824, buf825, 392, 512, grid=grid(392), stream=stream0)
        buf827 = empty((8, 512), device='cuda', dtype=torch.float32)
        buf828 = buf827; del buf827  # reuse
        # Source Nodes: [x_473, x_475, x_476], Original ATen: [aten.add, aten.mean, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_71.run(buf828, buf816, buf823, arg515_1, buf824, buf825, arg516_1, arg517_1, 4096, 49, grid=grid(4096), stream=stream0)
        del arg515_1
        del arg516_1
        del arg517_1
        del buf816
        del buf823
        del buf824
        del buf825
        buf829 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_473, x_475, x_476, x_478], Original ATen: [aten.add, aten.addmm, aten.mean, aten.native_layer_norm]
        extern_kernels.addmm(arg519_1, buf828, reinterpret_tensor(arg518_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf829)
        del arg518_1
        del arg519_1
        return (buf829, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
