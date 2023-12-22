
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
# Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
# l__mod___patch_embed_proj_0_0 => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/jn/cjn7owsoc72imajm47qpzi3mjlvbo3cgn6zecvjdwtxmzodmzmwr.py
# Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
# l__mod___patch_embed_proj_0_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3mqeoetr3yvnszk5glthoduei7fzvb575gkqkbp4oidunj4bg3.py
# Source Nodes: [l__mod___patch_embed_proj_0_1, l__mod___patch_embed_proj_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# l__mod___patch_embed_proj_0_1 => add_1, mul_1, mul_2, sub
# l__mod___patch_embed_proj_1 => add_2, erf, mul_3, mul_4, mul_5
triton_poi_fused__native_batch_norm_legit_no_training_gelu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 12544
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
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = tmp19 + tmp8
    tmp21 = tmp16 * tmp20
    tl.store(out_ptr0 + (y0 + (192*x2) + (2408448*y1)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmgiplo3ve46p3eb4ahdyua4oqm5aaiv2t3c4cxnnm6a6qtqhix.py
# Source Nodes: [l__mod___patch_embed_proj_1, l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution, aten.gelu]
# l__mod___patch_embed_proj_1 => add_2, erf, mul_3, mul_4, mul_5
# l__mod___patch_embed_proj_2_0 => convolution_1
triton_poi_fused_convolution_gelu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosgt76a4kdgttlxguyqa6gtiterblvmw6ownog44uxu7wcial3c.py
# Source Nodes: [l__mod___patch_embed_proj_2_1, l__mod___patch_embed_proj_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# l__mod___patch_embed_proj_2_1 => add_4, mul_7, mul_8, sub_1
# l__mod___patch_embed_proj_3 => add_5, erf_1, mul_10, mul_11, mul_9
triton_poi_fused__native_batch_norm_legit_no_training_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
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
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = tmp19 + tmp8
    tmp21 = tmp16 * tmp20
    tl.store(out_ptr0 + (y0 + (384*x2) + (1204224*y1)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clycwvxs4hxvropagtt2yyhviyaisnjl3ng5skxcufc35ozljqgc.py
# Source Nodes: [l__mod___patch_embed_proj_3, l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution, aten.gelu]
# l__mod___patch_embed_proj_3 => add_5, erf_1, mul_10, mul_11, mul_9
# l__mod___patch_embed_proj_4_0 => convolution_2
triton_poi_fused_convolution_gelu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ga/cga4wqqgwltylk42yplxjwgwtjxdutaso5l4uny3msz4v5y5wtth.py
# Source Nodes: [stack_2], Original ATen: [aten.stack]
# stack_2 => cat_1
triton_poi_fused_stack_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 896)
    x1 = (xindex // 2) % 16
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp7
    tmp10 = 28.000001907348633
    tmp11 = tmp9 / tmp10
    tmp12 = 6.283185307179586
    tmp13 = tmp11 * tmp12
    tmp14 = 2*x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp7
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = 2.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.math.floor(tmp20)
    tmp22 = tmp21 * tmp19
    tmp23 = 32.0
    tmp24 = tmp22 / tmp23
    tmp25 = 10000.0
    tmp26 = tl.math.pow(tmp25, tmp24)
    tmp27 = tmp13 / tmp26
    tmp28 = tl.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = 1 + (2*x1)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp7
    tmp37 = tmp36 + tmp17
    tmp38 = tmp37 / tmp19
    tmp39 = tl.math.floor(tmp38)
    tmp40 = tmp39 * tmp19
    tmp41 = tmp40 / tmp23
    tmp42 = tl.math.pow(tmp25, tmp41)
    tmp43 = tmp13 / tmp42
    tmp44 = tl.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp30, tmp46)
    tl.store(out_ptr0 + (x5), tmp47, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogmebiar7yskdmxq72jq6chw2pkp2xbujs7xklbt2pi5oxngf3g.py
# Source Nodes: [stack_3], Original ATen: [aten.stack]
# stack_3 => cat
triton_poi_fused_stack_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x2 = (xindex // 32) % 28
    x1 = (xindex // 2) % 16
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp7
    tmp10 = 28.000001907348633
    tmp11 = tmp9 / tmp10
    tmp12 = 6.283185307179586
    tmp13 = tmp11 * tmp12
    tmp14 = 2*x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp7
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = 2.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.math.floor(tmp20)
    tmp22 = tmp21 * tmp19
    tmp23 = 32.0
    tmp24 = tmp22 / tmp23
    tmp25 = 10000.0
    tmp26 = tl.math.pow(tmp25, tmp24)
    tmp27 = tmp13 / tmp26
    tmp28 = tl.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = 1 + (2*x1)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp7
    tmp37 = tmp36 + tmp17
    tmp38 = tmp37 / tmp19
    tmp39 = tl.math.floor(tmp38)
    tmp40 = tmp39 * tmp19
    tmp41 = tmp40 / tmp23
    tmp42 = tl.math.pow(tmp25, tmp41)
    tmp43 = tmp13 / tmp42
    tmp44 = tl.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp30, tmp46)
    tl.store(out_ptr0 + (x5), tmp47, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cok4te34m6htcknnlmqmxypt54rjdpsv376bbz6ztp4mx34djygq.py
# Source Nodes: [cat_11], Original ATen: [aten.cat]
# cat_11 => cat_2
triton_poi_fused_cat_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (32*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-32) + x0 + (32*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zafcmt5fnvyrelniy5glpfvsrknjp6kylfojavmougpyxcrp4p.py
# Source Nodes: [x_3], Original ATen: [aten.add]
# x_3 => add_13
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 768
    x4 = xindex % 602112
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce35t72vorggt5rkysaqhr6xyppyudasetutwl2xrdakgpk7itw2.py
# Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm1 => clone_1, var_mean
triton_red_fused_native_layer_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 784
    x2 = (xindex // 4704)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (100352*x0) + (602112*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (784*x0) + (4704*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (784*x0) + (4704*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (784*x0) + (4704*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb532gdow7ufxsypqac7vsa4krzjvrd5z4c6b6swwvyhpcjyfgv5.py
# Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm1 => clone_1, var_mean
triton_per_fused_native_layer_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (4704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (4704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (4704*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dj/cdje6e36esdfrfu7mdnttmivzo2e5i43gygqipnjegu2w7m5p7ga.py
# Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm1 => add_14, add_15, clone_1, mul_21, mul_22, rsqrt, sub_3, var_mean
triton_poi_fused_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7xjcsqgisfx7b7arfipejzywyby5szpbjdsrbefcz6uvn5uoa4.py
# Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
# q_1 => pow_2, sum_1
triton_red_fused_linalg_vector_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (258048*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzcismlotaww3khgixx433zn4lv5i5w4vg3zlueeaivtmwgdkhr.py
# Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
# q_1 => pow_2, sum_1
triton_per_fused_linalg_vector_norm_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (5376*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6krfpgiowoqicd6r523k4fvgihfoliktnagxmsk6rwcyu5ufe2.py
# Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
# k_1 => pow_4, sum_2
triton_red_fused_linalg_vector_norm_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (768 + x0), None, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (2304*r2) + (258048*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhkncfe7wrzkkelm2lp6ljognlj4bhlpb2j7zssmi2bqrn6ygrh.py
# Source Nodes: [matmul, q_1], Original ATen: [aten.clone, aten.div]
# matmul => clone_2
# q_1 => div_6
triton_poi_fused_clone_div_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_div_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2304*x2) + (1806336*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sqrt(tmp3)
    tmp5 = 1e-12
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clh7d23unondrfhpuukjjmaufamnrvadrpzjzpyiqi2gebealp6v.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_3
triton_poi_fused_clone_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632) % 16
    x3 = (xindex // 602112)
    x4 = (xindex // 37632)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (48*x2) + (2304*x1) + (1806336*x3)), None)
    tmp1 = tl.load(in_ptr1 + (768 + x0 + (48*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (48*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sqrt(tmp3)
    tmp5 = 1e-12
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x5), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxxjdpixqbyc74k4yjw3nolai5djxi2rvqqswkgspxu32h2p6sm.py
# Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
# attn => mul_23
# attn_1 => amax, div_8, exp, sub_4, sum_3
triton_per_fused__softmax_mul_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x1 = (xindex // 48) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (48*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r3 + (48*x4)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblfjkc3hhtqboghcfdrm25tdqrpnpc5kjxnhjjag4bz2bu3xwwx.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_5
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1536 + y0 + (2304*x2) + (1806336*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1536 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkffo54yjfw6amwd6ekaagbgbsxgkilapxbrowbpotupv7z57ah.py
# Source Nodes: [x_6], Original ATen: [aten.clone]
# x_6 => clone_6
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3f7bycj7ijcjlztybns7i4aoozcw3k4blmpgwsgicf73jksouwn.py
# Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_0_norm3 => clone_8, var_mean_1
# mul_4 => mul_24
# x_6 => add_16
# x_8 => add_17
triton_red_fused_add_mul_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 784
    x2 = (xindex // 4704)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (100352*x0) + (602112*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp1 * tmp4
        tmp6 = tmp0 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4hwgnknnic3nf6rclwdbegcewghlmth7otwhsfbh5nxrow2hco.py
# Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_0_norm3 => clone_8, var_mean_1
# mul_4 => mul_24
# x_6 => add_16
# x_8 => add_17
triton_per_fused_add_mul_native_layer_norm_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/w5/cw52gay6zc6tofkcootkpxlcwrw5chmz5t5ybkfajuqi4gbcziwi.py
# Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_0_norm3 => add_18, add_19, clone_8, mul_25, mul_26, rsqrt_1, sub_5, var_mean_1
# mul_4 => mul_24
# x_6 => add_16
# x_8 => add_17
triton_poi_fused_add_mul_native_layer_norm_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_layer_norm_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdsfabkwjvcwbpdprez5hn4isktx5x7rx57qgqfx6y23ihavpx2.py
# Source Nodes: [x_10, x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
# x_10 => convolution_4
# x_11 => add_20, erf_2, mul_27, mul_28, mul_29
# x_12 => add_22, mul_31, mul_32, sub_6
triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sqrt(tmp15)
    tmp17 = 1 / tmp16
    tmp18 = tmp17 * tmp8
    tmp19 = tmp12 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (y0 + (768*x2) + (602112*y1)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsqnno3xd4gwnkwmpd2fd3aukvkg4cxrkndfxldjf5azj4jsksi.py
# Source Nodes: [mul_4, mul_5, x_15, x_6, x_8], Original ATen: [aten.add, aten.mul]
# mul_4 => mul_24
# mul_5 => mul_33
# x_15 => add_23
# x_6 => add_16
# x_8 => add_17
triton_poi_fused_add_mul_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
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
    tmp0 = tl.load(in_out_ptr0 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (784*x2) + (602112*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdh5mbz7ocnzc747tfv7ziy3dkspap5uz2bz7hkduhcbek7456ut.py
# Source Nodes: [l__mod___blocks_0_norm2], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm2 => clone_9, var_mean_2
triton_red_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
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
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjcymyjfenlwnfrwwfaoskinb3lpfr55i2h2ibqhdtsawzzo6ew.py
# Source Nodes: [x_17], Original ATen: [aten.gelu]
# x_17 => add_26, erf_3, mul_36, mul_37, mul_38
triton_poi_fused_gelu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36ps43fu2z4z62nszaksijwko3lbjo6h27ktoy3p5l6k223rh5a.py
# Source Nodes: [l__mod___blocks_1_norm3, mul_6, mul_8, x_23, x_25, x_27], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_1_norm3 => add_32, add_33, clone_19, mul_44, mul_45, rsqrt_4, sub_10, var_mean_4
# mul_6 => mul_39
# mul_8 => mul_43
# x_23 => add_27
# x_25 => add_30
# x_27 => add_31
triton_per_fused_add_mul_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (602112*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckiu4l7xcrvbponlzwrk2fpvwngqw4zkwf2e2qtwbh6sn323pzld.py
# Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_1_norm2 => clone_20, var_mean_5
# mul_9 => mul_52
# x_34 => add_37
triton_red_fused_add_mul_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 6
    x1 = (xindex // 6) % 784
    x2 = (xindex // 4704)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (784*r3) + (100352*x0) + (602112*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp1 * tmp4
        tmp6 = tmp0 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
    tl.store(out_ptr2 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5yvpft2mrszzppadmvekhku7k7ipefjvzrchgkpoofkzuqte4r.py
# Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_1_norm2 => add_38, add_39, clone_20, mul_53, mul_54, rsqrt_5, sub_12, var_mean_5
# mul_9 => mul_52
# x_34 => add_37
triton_poi_fused_add_mul_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_layer_norm_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chx4zfcyqsudmjbgl5nhanlz4gfzqwf3diryyzbu4upbryi3pzuh.py
# Source Nodes: [l__mod___blocks_2_norm1, mul_10, mul_9, x_34, x_42], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_2_norm1 => add_42, add_43, clone_23, mul_59, mul_60, rsqrt_6, sub_13, var_mean_6
# mul_10 => mul_58
# mul_9 => mul_52
# x_34 => add_37
# x_42 => add_41
triton_per_fused_add_mul_native_layer_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (602112*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rrtceciqwsh7srtbsh4fjbncsd2uojnyjb4sqm33bveozhyeej.py
# Source Nodes: [l__mod___blocks_2_norm3, mul_12, x_44, x_46], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_2_norm3 => add_46, add_47, clone_30, mul_63, mul_64, rsqrt_7, sub_15, var_mean_7
# mul_12 => mul_62
# x_44 => add_44
# x_46 => add_45
triton_per_fused_add_mul_native_layer_norm_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv5phnl2uj5mmr57kdbr662jk7jydf24teewz2eyqehbtskv6je.py
# Source Nodes: [l__mod___blocks_2_norm2, mul_12, mul_13, x_44, x_46, x_53], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_2_norm2 => add_52, add_53, clone_31, mul_72, mul_73, rsqrt_8, sub_17, var_mean_8
# mul_12 => mul_62
# mul_13 => mul_71
# x_44 => add_44
# x_46 => add_45
# x_53 => add_51
triton_per_fused_add_mul_native_layer_norm_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr5 + (x0 + (784*r2) + (602112*x1)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rvoipyuqza4zsomkkikyzhgxvy7rrm6laespbii2imgq6ydyhe.py
# Source Nodes: [l__mod___blocks_3_norm3, mul_14, mul_16, x_61, x_63, x_65], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# l__mod___blocks_3_norm3 => add_60, add_61, clone_41, mul_82, mul_83, rsqrt_10, sub_20, var_mean_10
# mul_14 => mul_77
# mul_16 => mul_81
# x_61 => add_55
# x_63 => add_58
# x_65 => add_59
triton_per_fused_add_mul_native_layer_norm_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33pqtmxz3meia75vptruppg4h4jmxpwji5sdxxvtaftbgbqpvyj.py
# Source Nodes: [cat_10, x_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_10 => cat_3
# x_norm1 => add_350, add_351, mul_477, mul_478, rsqrt_72, sub_123, var_mean_72
triton_per_fused_cat_native_layer_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp50 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr9 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-768) + r2 + (768*x0) + (602112*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + ((784*r2) + (602112*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr6 + ((-768) + r2 + (768*x0) + (602112*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp20 = tl.load(in_ptr7 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp17 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp8, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp7, tmp25)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.full([1], 768, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 / tmp35
    tmp37 = tmp27 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp26 - tmp36
    tmp44 = 768.0
    tmp45 = tmp42 / tmp44
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp53, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxq3ymp5wbqwvihqw3lz5asjodrofq3wiltqnwewk2xjhvx6b72.py
# Source Nodes: [cat_9, mul_99, x_462, x_res], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
# cat_9 => cat_4
# mul_99 => mul_479
# x_462 => add_352
# x_res => add_353, add_354, mul_480, mul_481, rsqrt_73, sub_124, var_mean_73
triton_per_fused_add_cat_mul_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = x0
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp2 < tmp5
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp2 >= tmp5
    tmp11 = tl.full([1], 785, tl.int64)
    tmp12 = tmp2 < tmp11
    tmp13 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & tmp10 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp6, tmp9, tmp15)
    tmp17 = tmp1 * tmp16
    tmp18 = tmp0 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7qxag7yrjydeqzo777wexmnlpgxm2ownqazx4jihzcynq6kexk.py
# Source Nodes: [x_464, x_465], Original ATen: [aten.add, aten.gelu]
# x_464 => add_355
# x_465 => add_356, erf_50, mul_482, mul_483, mul_484
triton_poi_fused_add_gelu_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
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


# kernel path: /tmp/torchinductor_youkaichao/37/c37wlrsvjhxzlbibhv4ki3dtw4i2x2gqlvppiz3k5jh7hs7pr5gf.py
# Source Nodes: [cat_8, x_472, x_norm1_1], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_8 => cat_5
# x_472 => add_357
# x_norm1_1 => add_358, add_359, mul_486, mul_487, rsqrt_74, sub_125, var_mean_74
triton_per_fused_add_cat_native_layer_norm_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp1 >= tmp4
    tmp14 = tl.full([1], 785, tl.int64)
    tmp15 = tmp1 < tmp14
    tmp16 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp13 & xmask, other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp12, tmp18)
    tmp20 = tmp0 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 768.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp47, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegv65rv2qlteuuwqco5pf2t6ky4rzm77mi3xrvn4lhhmrs4txm6.py
# Source Nodes: [cat_7, cat_8, mul_101, x_472, x_473, x_res_1], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
# cat_7 => cat_6
# cat_8 => cat_5
# mul_101 => mul_488
# x_472 => add_357
# x_473 => add_360
# x_res_1 => add_361, add_362, mul_489, mul_490, rsqrt_75, sub_126, var_mean_75
triton_per_fused_add_cat_mul_native_layer_norm_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp1 >= tmp4
    tmp14 = tl.full([1], 785, tl.int64)
    tmp15 = tmp1 < tmp14
    tmp16 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp13 & xmask, other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp12, tmp18)
    tmp20 = tmp0 + tmp19
    tmp22 = tl.load(in_ptr5 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & tmp13 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp13, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp24, tmp27)
    tmp29 = tmp21 * tmp28
    tmp30 = tmp20 + tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tl.full([1], 768, tl.int32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 / tmp39
    tmp41 = tmp31 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp45 = tl.where(rmask & xmask, tmp43, 0)
    tmp46 = triton_helpers.promote_to_tensor(tl.sum(tmp45, 0))
    tmp47 = tmp30 - tmp40
    tmp48 = 768.0
    tmp49 = tmp46 / tmp48
    tmp50 = 1e-06
    tmp51 = tmp49 + tmp50
    tmp52 = tl.math.rsqrt(tmp51)
    tmp53 = tmp47 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp30, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp57, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp55g3p52ht35xn2pnvqjganrsz3rzy3gif5x4d2cshj7zji4f5f.py
# Source Nodes: [cat_6, x_483, x_485], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_6 => cat_7
# x_483 => add_365
# x_485 => var_mean_76
triton_per_fused_add_cat_native_layer_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp1 >= tmp4
    tmp14 = tl.full([1], 785, tl.int64)
    tmp15 = tmp1 < tmp14
    tmp16 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp13 & xmask, other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp12, tmp18)
    tmp20 = tmp0 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tl.store(out_ptr0 + (x3), tmp30, xmask)
    tl.store(out_ptr1 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlewi5zu3g62uiinqfgnvw7gf42dhlwaanvyrkjwducmmijqnhv.py
# Source Nodes: [x_487], Original ATen: [aten.clone]
# x_487 => clone_271
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*x1)), None)
    tmp20 = tl.load(in_ptr4 + (785*x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (785*x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp1 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp1 < tmp3
    tmp5 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr2 + (x2), tmp4, other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 >= tmp3
    tmp13 = tl.full([1], 785, tl.int64)
    tmp14 = tmp1 < tmp13
    tmp15 = tl.load(in_ptr0 + (x0 + (602880*x1)), tmp12, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp11, tmp17)
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 - tmp20
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp21 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, ), (1, ))
    assert_size_stride(arg1_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (192, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg102_1, (192, ), (1, ))
    assert_size_stride(arg103_1, (192, ), (1, ))
    assert_size_stride(arg104_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (2304, 768), (768, 1))
    assert_size_stride(arg115_1, (2304, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (768, 3072), (3072, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (2304, 768), (768, 1))
    assert_size_stride(arg135_1, (2304, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (3072, 768), (768, 1))
    assert_size_stride(arg149_1, (3072, ), (1, ))
    assert_size_stride(arg150_1, (768, 3072), (3072, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (2304, 768), (768, 1))
    assert_size_stride(arg155_1, (2304, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (3072, 768), (768, 1))
    assert_size_stride(arg169_1, (3072, ), (1, ))
    assert_size_stride(arg170_1, (768, 3072), (3072, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (2304, 768), (768, 1))
    assert_size_stride(arg175_1, (2304, ), (1, ))
    assert_size_stride(arg176_1, (768, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (3072, 768), (768, 1))
    assert_size_stride(arg189_1, (3072, ), (1, ))
    assert_size_stride(arg190_1, (768, 3072), (3072, 1))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (2304, 768), (768, 1))
    assert_size_stride(arg195_1, (2304, ), (1, ))
    assert_size_stride(arg196_1, (768, 768), (768, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (3072, 768), (768, 1))
    assert_size_stride(arg209_1, (3072, ), (1, ))
    assert_size_stride(arg210_1, (768, 3072), (3072, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (2304, 768), (768, 1))
    assert_size_stride(arg215_1, (2304, ), (1, ))
    assert_size_stride(arg216_1, (768, 768), (768, 1))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (768, ), (1, ))
    assert_size_stride(arg224_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (3072, 768), (768, 1))
    assert_size_stride(arg229_1, (3072, ), (1, ))
    assert_size_stride(arg230_1, (768, 3072), (3072, 1))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (2304, 768), (768, 1))
    assert_size_stride(arg235_1, (2304, ), (1, ))
    assert_size_stride(arg236_1, (768, 768), (768, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (3072, 768), (768, 1))
    assert_size_stride(arg249_1, (3072, ), (1, ))
    assert_size_stride(arg250_1, (768, 3072), (3072, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (2304, 768), (768, 1))
    assert_size_stride(arg255_1, (2304, ), (1, ))
    assert_size_stride(arg256_1, (768, 768), (768, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg261_1, (768, ), (1, ))
    assert_size_stride(arg262_1, (768, ), (1, ))
    assert_size_stride(arg263_1, (768, ), (1, ))
    assert_size_stride(arg264_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg265_1, (768, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (3072, 768), (768, 1))
    assert_size_stride(arg269_1, (3072, ), (1, ))
    assert_size_stride(arg270_1, (768, 3072), (3072, 1))
    assert_size_stride(arg271_1, (768, ), (1, ))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (2304, 768), (768, 1))
    assert_size_stride(arg275_1, (2304, ), (1, ))
    assert_size_stride(arg276_1, (768, 768), (768, 1))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (768, ), (1, ))
    assert_size_stride(arg280_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg281_1, (768, ), (1, ))
    assert_size_stride(arg282_1, (768, ), (1, ))
    assert_size_stride(arg283_1, (768, ), (1, ))
    assert_size_stride(arg284_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg285_1, (768, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (3072, 768), (768, 1))
    assert_size_stride(arg289_1, (3072, ), (1, ))
    assert_size_stride(arg290_1, (768, 3072), (3072, 1))
    assert_size_stride(arg291_1, (768, ), (1, ))
    assert_size_stride(arg292_1, (768, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (2304, 768), (768, 1))
    assert_size_stride(arg295_1, (2304, ), (1, ))
    assert_size_stride(arg296_1, (768, 768), (768, 1))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (768, ), (1, ))
    assert_size_stride(arg300_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg301_1, (768, ), (1, ))
    assert_size_stride(arg302_1, (768, ), (1, ))
    assert_size_stride(arg303_1, (768, ), (1, ))
    assert_size_stride(arg304_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg305_1, (768, ), (1, ))
    assert_size_stride(arg306_1, (768, ), (1, ))
    assert_size_stride(arg307_1, (768, ), (1, ))
    assert_size_stride(arg308_1, (3072, 768), (768, 1))
    assert_size_stride(arg309_1, (3072, ), (1, ))
    assert_size_stride(arg310_1, (768, 3072), (3072, 1))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (2304, 768), (768, 1))
    assert_size_stride(arg315_1, (2304, ), (1, ))
    assert_size_stride(arg316_1, (768, 768), (768, 1))
    assert_size_stride(arg317_1, (768, ), (1, ))
    assert_size_stride(arg318_1, (768, ), (1, ))
    assert_size_stride(arg319_1, (768, ), (1, ))
    assert_size_stride(arg320_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg321_1, (768, ), (1, ))
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg325_1, (768, ), (1, ))
    assert_size_stride(arg326_1, (768, ), (1, ))
    assert_size_stride(arg327_1, (768, ), (1, ))
    assert_size_stride(arg328_1, (3072, 768), (768, 1))
    assert_size_stride(arg329_1, (3072, ), (1, ))
    assert_size_stride(arg330_1, (768, 3072), (3072, 1))
    assert_size_stride(arg331_1, (768, ), (1, ))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (768, ), (1, ))
    assert_size_stride(arg334_1, (2304, 768), (768, 1))
    assert_size_stride(arg335_1, (2304, ), (1, ))
    assert_size_stride(arg336_1, (768, 768), (768, 1))
    assert_size_stride(arg337_1, (768, ), (1, ))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (768, ), (1, ))
    assert_size_stride(arg340_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg341_1, (768, ), (1, ))
    assert_size_stride(arg342_1, (768, ), (1, ))
    assert_size_stride(arg343_1, (768, ), (1, ))
    assert_size_stride(arg344_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg345_1, (768, ), (1, ))
    assert_size_stride(arg346_1, (768, ), (1, ))
    assert_size_stride(arg347_1, (768, ), (1, ))
    assert_size_stride(arg348_1, (3072, 768), (768, 1))
    assert_size_stride(arg349_1, (3072, ), (1, ))
    assert_size_stride(arg350_1, (768, 3072), (3072, 1))
    assert_size_stride(arg351_1, (768, ), (1, ))
    assert_size_stride(arg352_1, (768, ), (1, ))
    assert_size_stride(arg353_1, (768, ), (1, ))
    assert_size_stride(arg354_1, (2304, 768), (768, 1))
    assert_size_stride(arg355_1, (2304, ), (1, ))
    assert_size_stride(arg356_1, (768, 768), (768, 1))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (768, ), (1, ))
    assert_size_stride(arg360_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (768, ), (1, ))
    assert_size_stride(arg363_1, (768, ), (1, ))
    assert_size_stride(arg364_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (768, ), (1, ))
    assert_size_stride(arg368_1, (3072, 768), (768, 1))
    assert_size_stride(arg369_1, (3072, ), (1, ))
    assert_size_stride(arg370_1, (768, 3072), (3072, 1))
    assert_size_stride(arg371_1, (768, ), (1, ))
    assert_size_stride(arg372_1, (768, ), (1, ))
    assert_size_stride(arg373_1, (768, ), (1, ))
    assert_size_stride(arg374_1, (2304, 768), (768, 1))
    assert_size_stride(arg375_1, (2304, ), (1, ))
    assert_size_stride(arg376_1, (768, 768), (768, 1))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (768, ), (1, ))
    assert_size_stride(arg379_1, (768, ), (1, ))
    assert_size_stride(arg380_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg381_1, (768, ), (1, ))
    assert_size_stride(arg382_1, (768, ), (1, ))
    assert_size_stride(arg383_1, (768, ), (1, ))
    assert_size_stride(arg384_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg385_1, (768, ), (1, ))
    assert_size_stride(arg386_1, (768, ), (1, ))
    assert_size_stride(arg387_1, (768, ), (1, ))
    assert_size_stride(arg388_1, (3072, 768), (768, 1))
    assert_size_stride(arg389_1, (3072, ), (1, ))
    assert_size_stride(arg390_1, (768, 3072), (3072, 1))
    assert_size_stride(arg391_1, (768, ), (1, ))
    assert_size_stride(arg392_1, (768, ), (1, ))
    assert_size_stride(arg393_1, (768, ), (1, ))
    assert_size_stride(arg394_1, (2304, 768), (768, 1))
    assert_size_stride(arg395_1, (2304, ), (1, ))
    assert_size_stride(arg396_1, (768, 768), (768, 1))
    assert_size_stride(arg397_1, (768, ), (1, ))
    assert_size_stride(arg398_1, (768, ), (1, ))
    assert_size_stride(arg399_1, (768, ), (1, ))
    assert_size_stride(arg400_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg401_1, (768, ), (1, ))
    assert_size_stride(arg402_1, (768, ), (1, ))
    assert_size_stride(arg403_1, (768, ), (1, ))
    assert_size_stride(arg404_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg405_1, (768, ), (1, ))
    assert_size_stride(arg406_1, (768, ), (1, ))
    assert_size_stride(arg407_1, (768, ), (1, ))
    assert_size_stride(arg408_1, (3072, 768), (768, 1))
    assert_size_stride(arg409_1, (3072, ), (1, ))
    assert_size_stride(arg410_1, (768, 3072), (3072, 1))
    assert_size_stride(arg411_1, (768, ), (1, ))
    assert_size_stride(arg412_1, (768, ), (1, ))
    assert_size_stride(arg413_1, (768, ), (1, ))
    assert_size_stride(arg414_1, (2304, 768), (768, 1))
    assert_size_stride(arg415_1, (2304, ), (1, ))
    assert_size_stride(arg416_1, (768, 768), (768, 1))
    assert_size_stride(arg417_1, (768, ), (1, ))
    assert_size_stride(arg418_1, (768, ), (1, ))
    assert_size_stride(arg419_1, (768, ), (1, ))
    assert_size_stride(arg420_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg421_1, (768, ), (1, ))
    assert_size_stride(arg422_1, (768, ), (1, ))
    assert_size_stride(arg423_1, (768, ), (1, ))
    assert_size_stride(arg424_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg425_1, (768, ), (1, ))
    assert_size_stride(arg426_1, (768, ), (1, ))
    assert_size_stride(arg427_1, (768, ), (1, ))
    assert_size_stride(arg428_1, (3072, 768), (768, 1))
    assert_size_stride(arg429_1, (3072, ), (1, ))
    assert_size_stride(arg430_1, (768, 3072), (3072, 1))
    assert_size_stride(arg431_1, (768, ), (1, ))
    assert_size_stride(arg432_1, (768, ), (1, ))
    assert_size_stride(arg433_1, (768, ), (1, ))
    assert_size_stride(arg434_1, (2304, 768), (768, 1))
    assert_size_stride(arg435_1, (2304, ), (1, ))
    assert_size_stride(arg436_1, (768, 768), (768, 1))
    assert_size_stride(arg437_1, (768, ), (1, ))
    assert_size_stride(arg438_1, (768, ), (1, ))
    assert_size_stride(arg439_1, (768, ), (1, ))
    assert_size_stride(arg440_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg441_1, (768, ), (1, ))
    assert_size_stride(arg442_1, (768, ), (1, ))
    assert_size_stride(arg443_1, (768, ), (1, ))
    assert_size_stride(arg444_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg445_1, (768, ), (1, ))
    assert_size_stride(arg446_1, (768, ), (1, ))
    assert_size_stride(arg447_1, (768, ), (1, ))
    assert_size_stride(arg448_1, (3072, 768), (768, 1))
    assert_size_stride(arg449_1, (3072, ), (1, ))
    assert_size_stride(arg450_1, (768, 3072), (3072, 1))
    assert_size_stride(arg451_1, (768, ), (1, ))
    assert_size_stride(arg452_1, (768, ), (1, ))
    assert_size_stride(arg453_1, (768, ), (1, ))
    assert_size_stride(arg454_1, (2304, 768), (768, 1))
    assert_size_stride(arg455_1, (2304, ), (1, ))
    assert_size_stride(arg456_1, (768, 768), (768, 1))
    assert_size_stride(arg457_1, (768, ), (1, ))
    assert_size_stride(arg458_1, (768, ), (1, ))
    assert_size_stride(arg459_1, (768, ), (1, ))
    assert_size_stride(arg460_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg461_1, (768, ), (1, ))
    assert_size_stride(arg462_1, (768, ), (1, ))
    assert_size_stride(arg463_1, (768, ), (1, ))
    assert_size_stride(arg464_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg465_1, (768, ), (1, ))
    assert_size_stride(arg466_1, (768, ), (1, ))
    assert_size_stride(arg467_1, (768, ), (1, ))
    assert_size_stride(arg468_1, (3072, 768), (768, 1))
    assert_size_stride(arg469_1, (3072, ), (1, ))
    assert_size_stride(arg470_1, (768, 3072), (3072, 1))
    assert_size_stride(arg471_1, (768, ), (1, ))
    assert_size_stride(arg472_1, (768, ), (1, ))
    assert_size_stride(arg473_1, (768, ), (1, ))
    assert_size_stride(arg474_1, (2304, 768), (768, 1))
    assert_size_stride(arg475_1, (2304, ), (1, ))
    assert_size_stride(arg476_1, (768, 768), (768, 1))
    assert_size_stride(arg477_1, (768, ), (1, ))
    assert_size_stride(arg478_1, (768, ), (1, ))
    assert_size_stride(arg479_1, (768, ), (1, ))
    assert_size_stride(arg480_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg481_1, (768, ), (1, ))
    assert_size_stride(arg482_1, (768, ), (1, ))
    assert_size_stride(arg483_1, (768, ), (1, ))
    assert_size_stride(arg484_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg485_1, (768, ), (1, ))
    assert_size_stride(arg486_1, (768, ), (1, ))
    assert_size_stride(arg487_1, (768, ), (1, ))
    assert_size_stride(arg488_1, (3072, 768), (768, 1))
    assert_size_stride(arg489_1, (3072, ), (1, ))
    assert_size_stride(arg490_1, (768, 3072), (3072, 1))
    assert_size_stride(arg491_1, (768, ), (1, ))
    assert_size_stride(arg492_1, (768, ), (1, ))
    assert_size_stride(arg493_1, (768, ), (1, ))
    assert_size_stride(arg494_1, (2304, 768), (768, 1))
    assert_size_stride(arg495_1, (2304, ), (1, ))
    assert_size_stride(arg496_1, (768, 768), (768, 1))
    assert_size_stride(arg497_1, (768, ), (1, ))
    assert_size_stride(arg498_1, (768, ), (1, ))
    assert_size_stride(arg499_1, (768, ), (1, ))
    assert_size_stride(arg500_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg501_1, (768, ), (1, ))
    assert_size_stride(arg502_1, (768, ), (1, ))
    assert_size_stride(arg503_1, (768, ), (1, ))
    assert_size_stride(arg504_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg505_1, (768, ), (1, ))
    assert_size_stride(arg506_1, (768, ), (1, ))
    assert_size_stride(arg507_1, (768, ), (1, ))
    assert_size_stride(arg508_1, (3072, 768), (768, 1))
    assert_size_stride(arg509_1, (3072, ), (1, ))
    assert_size_stride(arg510_1, (768, 3072), (3072, 1))
    assert_size_stride(arg511_1, (768, ), (1, ))
    assert_size_stride(arg512_1, (768, ), (1, ))
    assert_size_stride(arg513_1, (768, ), (1, ))
    assert_size_stride(arg514_1, (2304, 768), (768, 1))
    assert_size_stride(arg515_1, (2304, ), (1, ))
    assert_size_stride(arg516_1, (768, 768), (768, 1))
    assert_size_stride(arg517_1, (768, ), (1, ))
    assert_size_stride(arg518_1, (768, ), (1, ))
    assert_size_stride(arg519_1, (768, ), (1, ))
    assert_size_stride(arg520_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg521_1, (768, ), (1, ))
    assert_size_stride(arg522_1, (768, ), (1, ))
    assert_size_stride(arg523_1, (768, ), (1, ))
    assert_size_stride(arg524_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg525_1, (768, ), (1, ))
    assert_size_stride(arg526_1, (768, ), (1, ))
    assert_size_stride(arg527_1, (768, ), (1, ))
    assert_size_stride(arg528_1, (3072, 768), (768, 1))
    assert_size_stride(arg529_1, (3072, ), (1, ))
    assert_size_stride(arg530_1, (768, 3072), (3072, 1))
    assert_size_stride(arg531_1, (768, ), (1, ))
    assert_size_stride(arg532_1, (768, ), (1, ))
    assert_size_stride(arg533_1, (768, ), (1, ))
    assert_size_stride(arg534_1, (2304, 768), (768, 1))
    assert_size_stride(arg535_1, (2304, ), (1, ))
    assert_size_stride(arg536_1, (768, 768), (768, 1))
    assert_size_stride(arg537_1, (768, ), (1, ))
    assert_size_stride(arg538_1, (768, ), (1, ))
    assert_size_stride(arg539_1, (768, ), (1, ))
    assert_size_stride(arg540_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg541_1, (768, ), (1, ))
    assert_size_stride(arg542_1, (768, ), (1, ))
    assert_size_stride(arg543_1, (768, ), (1, ))
    assert_size_stride(arg544_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg545_1, (768, ), (1, ))
    assert_size_stride(arg546_1, (768, ), (1, ))
    assert_size_stride(arg547_1, (768, ), (1, ))
    assert_size_stride(arg548_1, (3072, 768), (768, 1))
    assert_size_stride(arg549_1, (3072, ), (1, ))
    assert_size_stride(arg550_1, (768, 3072), (3072, 1))
    assert_size_stride(arg551_1, (768, ), (1, ))
    assert_size_stride(arg552_1, (768, ), (1, ))
    assert_size_stride(arg553_1, (768, ), (1, ))
    assert_size_stride(arg554_1, (2304, 768), (768, 1))
    assert_size_stride(arg555_1, (2304, ), (1, ))
    assert_size_stride(arg556_1, (768, 768), (768, 1))
    assert_size_stride(arg557_1, (768, ), (1, ))
    assert_size_stride(arg558_1, (768, ), (1, ))
    assert_size_stride(arg559_1, (768, ), (1, ))
    assert_size_stride(arg560_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg561_1, (768, ), (1, ))
    assert_size_stride(arg562_1, (768, ), (1, ))
    assert_size_stride(arg563_1, (768, ), (1, ))
    assert_size_stride(arg564_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg565_1, (768, ), (1, ))
    assert_size_stride(arg566_1, (768, ), (1, ))
    assert_size_stride(arg567_1, (768, ), (1, ))
    assert_size_stride(arg568_1, (3072, 768), (768, 1))
    assert_size_stride(arg569_1, (3072, ), (1, ))
    assert_size_stride(arg570_1, (768, 3072), (3072, 1))
    assert_size_stride(arg571_1, (768, ), (1, ))
    assert_size_stride(arg572_1, (768, ), (1, ))
    assert_size_stride(arg573_1, (768, ), (1, ))
    assert_size_stride(arg574_1, (2304, 768), (768, 1))
    assert_size_stride(arg575_1, (2304, ), (1, ))
    assert_size_stride(arg576_1, (768, 768), (768, 1))
    assert_size_stride(arg577_1, (768, ), (1, ))
    assert_size_stride(arg578_1, (768, ), (1, ))
    assert_size_stride(arg579_1, (768, ), (1, ))
    assert_size_stride(arg580_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg581_1, (768, ), (1, ))
    assert_size_stride(arg582_1, (768, ), (1, ))
    assert_size_stride(arg583_1, (768, ), (1, ))
    assert_size_stride(arg584_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg585_1, (768, ), (1, ))
    assert_size_stride(arg586_1, (768, ), (1, ))
    assert_size_stride(arg587_1, (768, ), (1, ))
    assert_size_stride(arg588_1, (3072, 768), (768, 1))
    assert_size_stride(arg589_1, (3072, ), (1, ))
    assert_size_stride(arg590_1, (768, 3072), (3072, 1))
    assert_size_stride(arg591_1, (768, ), (1, ))
    assert_size_stride(arg592_1, (768, ), (1, ))
    assert_size_stride(arg593_1, (768, ), (1, ))
    assert_size_stride(arg594_1, (768, 768), (768, 1))
    assert_size_stride(arg595_1, (768, ), (1, ))
    assert_size_stride(arg596_1, (768, 768), (768, 1))
    assert_size_stride(arg597_1, (768, ), (1, ))
    assert_size_stride(arg598_1, (768, 768), (768, 1))
    assert_size_stride(arg599_1, (768, ), (1, ))
    assert_size_stride(arg600_1, (768, 768), (768, 1))
    assert_size_stride(arg601_1, (768, ), (1, ))
    assert_size_stride(arg602_1, (768, ), (1, ))
    assert_size_stride(arg603_1, (768, ), (1, ))
    assert_size_stride(arg604_1, (3072, 768), (768, 1))
    assert_size_stride(arg605_1, (3072, ), (1, ))
    assert_size_stride(arg606_1, (768, 3072), (3072, 1))
    assert_size_stride(arg607_1, (768, ), (1, ))
    assert_size_stride(arg608_1, (768, ), (1, ))
    assert_size_stride(arg609_1, (768, ), (1, ))
    assert_size_stride(arg610_1, (768, 768), (768, 1))
    assert_size_stride(arg611_1, (768, ), (1, ))
    assert_size_stride(arg612_1, (768, 768), (768, 1))
    assert_size_stride(arg613_1, (768, ), (1, ))
    assert_size_stride(arg614_1, (768, 768), (768, 1))
    assert_size_stride(arg615_1, (768, ), (1, ))
    assert_size_stride(arg616_1, (768, 768), (768, 1))
    assert_size_stride(arg617_1, (768, ), (1, ))
    assert_size_stride(arg618_1, (768, ), (1, ))
    assert_size_stride(arg619_1, (768, ), (1, ))
    assert_size_stride(arg620_1, (3072, 768), (768, 1))
    assert_size_stride(arg621_1, (3072, ), (1, ))
    assert_size_stride(arg622_1, (768, 3072), (3072, 1))
    assert_size_stride(arg623_1, (768, ), (1, ))
    assert_size_stride(arg624_1, (768, ), (1, ))
    assert_size_stride(arg625_1, (768, ), (1, ))
    assert_size_stride(arg626_1, (1000, 768), (768, 1))
    assert_size_stride(arg627_1, (1000, ), (1, ))
    assert_size_stride(arg628_1, (192, ), (1, ))
    assert_size_stride(arg629_1, (192, ), (1, ))
    assert_size_stride(arg630_1, (), ())
    assert_size_stride(arg631_1, (384, ), (1, ))
    assert_size_stride(arg632_1, (384, ), (1, ))
    assert_size_stride(arg633_1, (), ())
    assert_size_stride(arg634_1, (768, ), (1, ))
    assert_size_stride(arg635_1, (768, ), (1, ))
    assert_size_stride(arg636_1, (), ())
    assert_size_stride(arg637_1, (768, ), (1, ))
    assert_size_stride(arg638_1, (768, ), (1, ))
    assert_size_stride(arg639_1, (), ())
    assert_size_stride(arg640_1, (768, ), (1, ))
    assert_size_stride(arg641_1, (768, ), (1, ))
    assert_size_stride(arg642_1, (), ())
    assert_size_stride(arg643_1, (768, ), (1, ))
    assert_size_stride(arg644_1, (768, ), (1, ))
    assert_size_stride(arg645_1, (), ())
    assert_size_stride(arg646_1, (768, ), (1, ))
    assert_size_stride(arg647_1, (768, ), (1, ))
    assert_size_stride(arg648_1, (), ())
    assert_size_stride(arg649_1, (768, ), (1, ))
    assert_size_stride(arg650_1, (768, ), (1, ))
    assert_size_stride(arg651_1, (), ())
    assert_size_stride(arg652_1, (768, ), (1, ))
    assert_size_stride(arg653_1, (768, ), (1, ))
    assert_size_stride(arg654_1, (), ())
    assert_size_stride(arg655_1, (768, ), (1, ))
    assert_size_stride(arg656_1, (768, ), (1, ))
    assert_size_stride(arg657_1, (), ())
    assert_size_stride(arg658_1, (768, ), (1, ))
    assert_size_stride(arg659_1, (768, ), (1, ))
    assert_size_stride(arg660_1, (), ())
    assert_size_stride(arg661_1, (768, ), (1, ))
    assert_size_stride(arg662_1, (768, ), (1, ))
    assert_size_stride(arg663_1, (), ())
    assert_size_stride(arg664_1, (768, ), (1, ))
    assert_size_stride(arg665_1, (768, ), (1, ))
    assert_size_stride(arg666_1, (), ())
    assert_size_stride(arg667_1, (768, ), (1, ))
    assert_size_stride(arg668_1, (768, ), (1, ))
    assert_size_stride(arg669_1, (), ())
    assert_size_stride(arg670_1, (768, ), (1, ))
    assert_size_stride(arg671_1, (768, ), (1, ))
    assert_size_stride(arg672_1, (), ())
    assert_size_stride(arg673_1, (768, ), (1, ))
    assert_size_stride(arg674_1, (768, ), (1, ))
    assert_size_stride(arg675_1, (), ())
    assert_size_stride(arg676_1, (768, ), (1, ))
    assert_size_stride(arg677_1, (768, ), (1, ))
    assert_size_stride(arg678_1, (), ())
    assert_size_stride(arg679_1, (768, ), (1, ))
    assert_size_stride(arg680_1, (768, ), (1, ))
    assert_size_stride(arg681_1, (), ())
    assert_size_stride(arg682_1, (768, ), (1, ))
    assert_size_stride(arg683_1, (768, ), (1, ))
    assert_size_stride(arg684_1, (), ())
    assert_size_stride(arg685_1, (768, ), (1, ))
    assert_size_stride(arg686_1, (768, ), (1, ))
    assert_size_stride(arg687_1, (), ())
    assert_size_stride(arg688_1, (768, ), (1, ))
    assert_size_stride(arg689_1, (768, ), (1, ))
    assert_size_stride(arg690_1, (), ())
    assert_size_stride(arg691_1, (768, ), (1, ))
    assert_size_stride(arg692_1, (768, ), (1, ))
    assert_size_stride(arg693_1, (), ())
    assert_size_stride(arg694_1, (768, ), (1, ))
    assert_size_stride(arg695_1, (768, ), (1, ))
    assert_size_stride(arg696_1, (), ())
    assert_size_stride(arg697_1, (768, ), (1, ))
    assert_size_stride(arg698_1, (768, ), (1, ))
    assert_size_stride(arg699_1, (), ())
    assert_size_stride(arg700_1, (768, ), (1, ))
    assert_size_stride(arg701_1, (768, ), (1, ))
    assert_size_stride(arg702_1, (), ())
    assert_size_stride(arg703_1, (768, ), (1, ))
    assert_size_stride(arg704_1, (768, ), (1, ))
    assert_size_stride(arg705_1, (), ())
    assert_size_stride(arg706_1, (768, ), (1, ))
    assert_size_stride(arg707_1, (768, ), (1, ))
    assert_size_stride(arg708_1, (), ())
    assert_size_stride(arg709_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg709_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg709_1
        buf1 = empty_strided((192, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg101_1, buf1, 576, 9, grid=grid(576, 9), stream=stream0)
        del arg101_1
        # Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 192, 112, 112), (2408448, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_1, l__mod___patch_embed_proj_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_gelu_2.run(buf3, arg628_1, arg629_1, arg102_1, arg103_1, buf4, 1536, 12544, grid=grid(1536, 12544), stream=stream0)
        del arg102_1
        del arg103_1
        del arg628_1
        del arg629_1
        del buf3
        buf5 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_1, l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_3.run(arg104_1, buf5, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg104_1
        # Source Nodes: [l__mod___patch_embed_proj_1, l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution, aten.gelu]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_2_1, l__mod___patch_embed_proj_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_gelu_4.run(buf7, arg631_1, arg632_1, arg105_1, arg106_1, buf8, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg105_1
        del arg106_1
        del arg631_1
        del arg632_1
        del buf7
        buf9 = empty_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_3, l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_5.run(arg107_1, buf9, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del arg107_1
        # Source Nodes: [l__mod___patch_embed_proj_3, l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution, aten.gelu]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 768, 28, 28), (602112, 784, 28, 1))
        del buf8
        del buf9
        buf11 = empty((1, 28, 28, 16, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_2], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf11, 25088, grid=grid(25088), stream=stream0)
        buf12 = empty((1, 28, 28, 16, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_3], Original ATen: [aten.stack]
        triton_poi_fused_stack_7.run(buf12, 25088, grid=grid(25088), stream=stream0)
        buf13 = empty((1, 28, 28, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf11, buf12, buf13, 50176, grid=grid(50176), stream=stream0)
        del buf11
        del buf12
        # Source Nodes: [pos_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (1, 64, 28, 28), (0, 1, 1792, 64), 0), arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (1, 768, 28, 28), (602112, 784, 28, 1))
        del arg110_1
        del buf13
        buf15 = reinterpret_tensor(buf10, (8, 784, 768), (602112, 1, 784), 0); del buf10  # reuse
        # Source Nodes: [x_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf15, arg634_1, arg635_1, arg108_1, arg109_1, buf14, arg111_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg108_1
        del arg109_1
        del arg111_1
        del arg634_1
        del arg635_1
        del buf14
        buf16 = empty_strided((8, 784, 1, 6), (4704, 1, 37632, 784), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((8, 784, 1, 6), (4704, 1, 37632, 784), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((8, 784, 1, 6), (4704, 1, 37632, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf15, buf16, buf17, buf18, 37632, 128, grid=grid(37632), stream=stream0)
        buf19 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_11.run(buf16, buf17, buf18, buf19, buf20, 6272, 6, grid=grid(6272), stream=stream0)
        buf22 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_12.run(buf15, buf19, buf20, arg112_1, arg113_1, buf22, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg112_1
        del arg113_1
        buf23 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (6272, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 2304), (1, 768), 0), out=buf23)
        del arg114_1
        buf24 = empty_strided((8, 16, 48, 1, 7), (5376, 48, 1, 43008, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf23, arg115_1, buf24, 43008, 112, grid=grid(43008), stream=stream0)
        buf25 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf24, buf25, 6144, 7, grid=grid(6144), stream=stream0)
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf23, arg115_1, buf26, 43008, 112, grid=grid(43008), stream=stream0)
        buf27 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf26, buf27, 6144, 7, grid=grid(6144), stream=stream0)
        buf28 = reinterpret_tensor(buf22, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf22  # reuse
        # Source Nodes: [matmul, q_1], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf23, arg115_1, buf25, buf28, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf29 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf23, arg115_1, buf27, buf29, 4816896, grid=grid(4816896), stream=stream0)
        buf30 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf29, (128, 784, 48), (37632, 48, 1), 0), out=buf30)
        buf33 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf30, arg1_1, buf33, 6144, 48, grid=grid(6144), stream=stream0)
        del arg1_1
        buf34 = reinterpret_tensor(buf29, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf29  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf23, arg115_1, buf34, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg115_1
        buf35 = reinterpret_tensor(buf28, (128, 48, 784), (37632, 784, 1), 0); del buf28  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf34, (128, 48, 784), (37632, 784, 1), 0), out=buf35)
        buf36 = reinterpret_tensor(buf34, (8, 784, 768), (602112, 768, 1), 0); del buf34  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf35, buf36, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (6272, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (6272, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf37)
        del arg116_1
        buf38 = reinterpret_tensor(buf18, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf18  # reuse
        buf39 = reinterpret_tensor(buf17, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf17  # reuse
        buf40 = reinterpret_tensor(buf16, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf16  # reuse
        # Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_21.run(buf15, arg0_1, buf37, arg117_1, buf38, buf39, buf40, 37632, 128, grid=grid(37632), stream=stream0)
        buf41 = buf20; del buf20  # reuse
        buf42 = buf19; del buf19  # reuse
        # Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf38, buf39, buf40, buf41, buf42, 6272, 6, grid=grid(6272), stream=stream0)
        buf44 = buf36; del buf36  # reuse
        # Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_23.run(buf15, arg0_1, buf37, arg117_1, buf41, buf42, arg118_1, arg119_1, buf44, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg118_1
        del arg119_1
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(reinterpret_tensor(buf44, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf45, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg120_1
        buf46 = reinterpret_tensor(buf44, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf44  # reuse
        # Source Nodes: [x_10, x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf45, arg121_1, arg637_1, arg638_1, arg122_1, arg123_1, buf46, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg637_1
        del arg638_1
        del buf45
        # Source Nodes: [x_10, x_11, x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf47 = extern_kernels.convolution(buf46, arg124_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf47, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg124_1
        buf48 = buf15; del buf15  # reuse
        # Source Nodes: [mul_4, mul_5, x_15, x_6, x_8], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf48, arg0_1, buf37, arg117_1, arg2_1, buf47, arg125_1, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg0_1
        del arg117_1
        del arg125_1
        del arg2_1
        buf49 = reinterpret_tensor(buf40, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf40  # reuse
        buf50 = reinterpret_tensor(buf39, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf39  # reuse
        buf51 = reinterpret_tensor(buf38, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf38  # reuse
        # Source Nodes: [l__mod___blocks_0_norm2], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_26.run(buf48, buf49, buf50, buf51, 37632, 128, grid=grid(37632), stream=stream0)
        buf52 = buf42; del buf42  # reuse
        buf53 = buf41; del buf41  # reuse
        # Source Nodes: [l__mod___blocks_0_norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_11.run(buf49, buf50, buf51, buf52, buf53, 6272, 6, grid=grid(6272), stream=stream0)
        buf55 = reinterpret_tensor(buf47, (8, 784, 768), (602112, 768, 1), 0); del buf47  # reuse
        # Source Nodes: [l__mod___blocks_0_norm2], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_12.run(buf48, buf52, buf53, arg126_1, arg127_1, buf55, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg126_1
        del arg127_1
        buf56 = reinterpret_tensor(buf4, (6272, 3072), (3072, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (6272, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 3072), (1, 768), 0), out=buf56)
        del arg128_1
        buf57 = reinterpret_tensor(buf56, (8, 784, 3072), (2408448, 3072, 1), 0); del buf56  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf57, arg129_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg129_1
        buf58 = reinterpret_tensor(buf55, (6272, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), out=buf58)
        del arg130_1
        buf59 = reinterpret_tensor(buf51, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf51  # reuse
        buf60 = reinterpret_tensor(buf50, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf50  # reuse
        buf61 = reinterpret_tensor(buf49, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf49  # reuse
        # Source Nodes: [l__mod___blocks_1_norm1, mul_6, x_23], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_21.run(buf48, arg3_1, buf58, arg131_1, buf59, buf60, buf61, 37632, 128, grid=grid(37632), stream=stream0)
        buf62 = buf53; del buf53  # reuse
        buf63 = buf52; del buf52  # reuse
        # Source Nodes: [l__mod___blocks_1_norm1, mul_6, x_23], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf59, buf60, buf61, buf62, buf63, 6272, 6, grid=grid(6272), stream=stream0)
        buf65 = reinterpret_tensor(buf37, (8, 784, 768), (602112, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [l__mod___blocks_1_norm1, mul_6, x_23], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_23.run(buf48, arg3_1, buf58, arg131_1, buf62, buf63, arg132_1, arg133_1, buf65, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg132_1
        del arg133_1
        buf66 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (6272, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 2304), (1, 768), 0), out=buf66)
        del arg134_1
        buf67 = buf26; del buf26  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf66, arg135_1, buf67, 43008, 112, grid=grid(43008), stream=stream0)
        buf68 = buf27; del buf27  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf67, buf68, 6144, 7, grid=grid(6144), stream=stream0)
        buf69 = buf67; del buf67  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf66, arg135_1, buf69, 43008, 112, grid=grid(43008), stream=stream0)
        buf70 = buf25; del buf25  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf69, buf70, 6144, 7, grid=grid(6144), stream=stream0)
        buf71 = reinterpret_tensor(buf65, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf65  # reuse
        # Source Nodes: [matmul_2, q_3], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf66, arg135_1, buf68, buf71, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf72 = reinterpret_tensor(buf46, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf46  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf66, arg135_1, buf70, buf72, 4816896, grid=grid(4816896), stream=stream0)
        buf73 = reinterpret_tensor(buf33, (128, 48, 48), (2304, 48, 1), 0); del buf33  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf72, (128, 784, 48), (37632, 48, 1), 0), out=buf73)
        buf76 = reinterpret_tensor(buf30, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf30  # reuse
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf73, arg5_1, buf76, 6144, 48, grid=grid(6144), stream=stream0)
        del arg5_1
        buf77 = reinterpret_tensor(buf72, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf72  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf66, arg135_1, buf77, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg135_1
        buf78 = reinterpret_tensor(buf71, (128, 48, 784), (37632, 784, 1), 0); del buf71  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf77, (128, 48, 784), (37632, 784, 1), 0), out=buf78)
        buf79 = reinterpret_tensor(buf77, (8, 784, 768), (602112, 768, 1), 0); del buf77  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf78, buf79, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf80 = reinterpret_tensor(buf78, (6272, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (6272, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), out=buf80)
        del arg136_1
        buf81 = reinterpret_tensor(buf58, (8, 784, 768), (602112, 768, 1), 0); del buf58  # reuse
        buf85 = buf79; del buf79  # reuse
        # Source Nodes: [l__mod___blocks_1_norm3, mul_6, mul_8, x_23, x_25, x_27], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_28.run(buf81, buf48, arg3_1, arg131_1, arg4_1, buf80, arg137_1, arg138_1, arg139_1, buf85, 6272, 768, grid=grid(6272), stream=stream0)
        del arg131_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg3_1
        del arg4_1
        del buf48
        del buf80
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg140_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf86, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg140_1
        buf87 = reinterpret_tensor(buf85, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf85  # reuse
        # Source Nodes: [x_29, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf86, arg141_1, arg640_1, arg641_1, arg142_1, arg143_1, buf87, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        del arg640_1
        del arg641_1
        # Source Nodes: [x_29, x_30, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf88 = extern_kernels.convolution(buf87, arg144_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf88, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg144_1
        buf89 = buf61; del buf61  # reuse
        buf90 = buf60; del buf60  # reuse
        buf91 = buf59; del buf59  # reuse
        # Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf81, arg6_1, buf88, arg145_1, buf89, buf90, buf91, 37632, 128, grid=grid(37632), stream=stream0)
        buf92 = buf63; del buf63  # reuse
        buf93 = buf62; del buf62  # reuse
        # Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf89, buf90, buf91, buf92, buf93, 6272, 6, grid=grid(6272), stream=stream0)
        buf95 = reinterpret_tensor(buf87, (8, 784, 768), (602112, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf81, arg6_1, buf88, arg145_1, buf92, buf93, arg146_1, arg147_1, buf95, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg146_1
        del arg147_1
        buf96 = reinterpret_tensor(buf57, (6272, 3072), (3072, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (6272, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 3072), (1, 768), 0), out=buf96)
        del arg148_1
        buf97 = reinterpret_tensor(buf96, (8, 784, 3072), (2408448, 3072, 1), 0); del buf96  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf97, arg149_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg149_1
        buf98 = reinterpret_tensor(buf95, (6272, 768), (768, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg150_1, (3072, 768), (1, 3072), 0), out=buf98)
        del arg150_1
        buf99 = reinterpret_tensor(buf98, (8, 784, 768), (602112, 768, 1), 0); del buf98  # reuse
        buf103 = reinterpret_tensor(buf86, (8, 784, 768), (602112, 768, 1), 0); del buf86  # reuse
        # Source Nodes: [l__mod___blocks_2_norm1, mul_10, mul_9, x_34, x_42], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf99, buf81, arg6_1, buf88, arg145_1, arg7_1, arg151_1, arg152_1, arg153_1, buf103, 6272, 768, grid=grid(6272), stream=stream0)
        del arg145_1
        del arg151_1
        del arg152_1
        del arg153_1
        del arg6_1
        del arg7_1
        del buf81
        buf104 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (6272, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 2304), (1, 768), 0), out=buf104)
        del arg154_1
        buf105 = buf69; del buf69  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf104, arg155_1, buf105, 43008, 112, grid=grid(43008), stream=stream0)
        buf106 = buf70; del buf70  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf105, buf106, 6144, 7, grid=grid(6144), stream=stream0)
        buf107 = buf105; del buf105  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf104, arg155_1, buf107, 43008, 112, grid=grid(43008), stream=stream0)
        buf108 = buf68; del buf68  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf107, buf108, 6144, 7, grid=grid(6144), stream=stream0)
        buf109 = reinterpret_tensor(buf103, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf103  # reuse
        # Source Nodes: [matmul_4, q_5], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf104, arg155_1, buf106, buf109, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf110 = reinterpret_tensor(buf88, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf88  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf104, arg155_1, buf108, buf110, 4816896, grid=grid(4816896), stream=stream0)
        buf111 = reinterpret_tensor(buf76, (128, 48, 48), (2304, 48, 1), 0); del buf76  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf110, (128, 784, 48), (37632, 48, 1), 0), out=buf111)
        buf114 = reinterpret_tensor(buf73, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf73  # reuse
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf111, arg9_1, buf114, 6144, 48, grid=grid(6144), stream=stream0)
        del arg9_1
        buf115 = reinterpret_tensor(buf110, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf110  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf104, arg155_1, buf115, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg155_1
        buf116 = reinterpret_tensor(buf109, (128, 48, 784), (37632, 784, 1), 0); del buf109  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf115, (128, 48, 784), (37632, 784, 1), 0), out=buf116)
        buf117 = reinterpret_tensor(buf115, (8, 784, 768), (602112, 768, 1), 0); del buf115  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf116, buf117, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf118 = reinterpret_tensor(buf116, (6272, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (6272, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), out=buf118)
        del arg156_1
        buf122 = buf117; del buf117  # reuse
        # Source Nodes: [l__mod___blocks_2_norm3, mul_12, x_44, x_46], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf99, arg8_1, buf118, arg157_1, arg158_1, arg159_1, buf122, 6272, 768, grid=grid(6272), stream=stream0)
        del arg158_1
        del arg159_1
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(reinterpret_tensor(buf122, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg160_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf123, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg160_1
        buf124 = reinterpret_tensor(buf122, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf122  # reuse
        # Source Nodes: [x_48, x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf123, arg161_1, arg643_1, arg644_1, arg162_1, arg163_1, buf124, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        del arg643_1
        del arg644_1
        # Source Nodes: [x_48, x_49, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf125 = extern_kernels.convolution(buf124, arg164_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf125, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg164_1
        buf126 = reinterpret_tensor(buf124, (8, 784, 768), (602112, 768, 1), 0); del buf124  # reuse
        buf130 = reinterpret_tensor(buf123, (8, 784, 768), (602112, 768, 1), 0); del buf123  # reuse
        # Source Nodes: [l__mod___blocks_2_norm2, mul_12, mul_13, x_44, x_46, x_53], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf99, arg8_1, buf118, arg157_1, arg10_1, buf125, arg165_1, arg166_1, arg167_1, buf126, buf130, 6272, 768, grid=grid(6272), stream=stream0)
        del arg10_1
        del arg157_1
        del arg165_1
        del arg166_1
        del arg167_1
        del arg8_1
        del buf118
        buf131 = reinterpret_tensor(buf97, (6272, 3072), (3072, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (6272, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 3072), (1, 768), 0), out=buf131)
        del arg168_1
        buf132 = reinterpret_tensor(buf131, (8, 784, 3072), (2408448, 3072, 1), 0); del buf131  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf132, arg169_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg169_1
        buf133 = reinterpret_tensor(buf130, (6272, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg170_1, (3072, 768), (1, 3072), 0), out=buf133)
        del arg170_1
        buf137 = buf99; del buf99  # reuse
        # Source Nodes: [l__mod___blocks_3_norm1, mul_14, x_61], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf126, arg11_1, buf133, arg171_1, arg172_1, arg173_1, buf137, 6272, 768, grid=grid(6272), stream=stream0)
        del arg172_1
        del arg173_1
        buf138 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (6272, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 2304), (1, 768), 0), out=buf138)
        del arg174_1
        buf139 = buf107; del buf107  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf138, arg175_1, buf139, 43008, 112, grid=grid(43008), stream=stream0)
        buf140 = buf108; del buf108  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf139, buf140, 6144, 7, grid=grid(6144), stream=stream0)
        buf141 = buf139; del buf139  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf138, arg175_1, buf141, 43008, 112, grid=grid(43008), stream=stream0)
        buf142 = buf106; del buf106  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf141, buf142, 6144, 7, grid=grid(6144), stream=stream0)
        buf143 = reinterpret_tensor(buf137, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf137  # reuse
        # Source Nodes: [matmul_6, q_7], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf138, arg175_1, buf140, buf143, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf144 = reinterpret_tensor(buf125, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf125  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf138, arg175_1, buf142, buf144, 4816896, grid=grid(4816896), stream=stream0)
        buf145 = reinterpret_tensor(buf114, (128, 48, 48), (2304, 48, 1), 0); del buf114  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf144, (128, 784, 48), (37632, 48, 1), 0), out=buf145)
        buf148 = reinterpret_tensor(buf111, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf111  # reuse
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf145, arg13_1, buf148, 6144, 48, grid=grid(6144), stream=stream0)
        del arg13_1
        buf149 = reinterpret_tensor(buf144, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf144  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf138, arg175_1, buf149, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg175_1
        buf150 = reinterpret_tensor(buf143, (128, 48, 784), (37632, 784, 1), 0); del buf143  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf149, (128, 48, 784), (37632, 784, 1), 0), out=buf150)
        buf151 = reinterpret_tensor(buf149, (8, 784, 768), (602112, 768, 1), 0); del buf149  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf150, buf151, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (6272, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (6272, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 768), (1, 768), 0), out=buf152)
        del arg176_1
        buf153 = reinterpret_tensor(buf152, (8, 784, 768), (602112, 768, 1), 0); del buf152  # reuse
        buf157 = buf151; del buf151  # reuse
        # Source Nodes: [l__mod___blocks_3_norm3, mul_14, mul_16, x_61, x_63, x_65], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf153, buf126, arg11_1, buf133, arg171_1, arg12_1, arg177_1, arg178_1, arg179_1, buf157, 6272, 768, grid=grid(6272), stream=stream0)
        del arg11_1
        del arg12_1
        del arg171_1
        del arg177_1
        del arg178_1
        del arg179_1
        del buf126
        del buf133
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(reinterpret_tensor(buf157, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg180_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf158, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg180_1
        buf159 = reinterpret_tensor(buf157, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf157  # reuse
        # Source Nodes: [x_67, x_68, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf158, arg181_1, arg646_1, arg647_1, arg182_1, arg183_1, buf159, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg181_1
        del arg182_1
        del arg183_1
        del arg646_1
        del arg647_1
        # Source Nodes: [x_67, x_68, x_69, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf160 = extern_kernels.convolution(buf159, arg184_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf160, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg184_1
        buf161 = buf91; del buf91  # reuse
        buf162 = buf90; del buf90  # reuse
        buf163 = buf89; del buf89  # reuse
        # Source Nodes: [l__mod___blocks_3_norm2, mul_17, x_72], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf153, arg14_1, buf160, arg185_1, buf161, buf162, buf163, 37632, 128, grid=grid(37632), stream=stream0)
        buf164 = buf93; del buf93  # reuse
        buf165 = buf92; del buf92  # reuse
        # Source Nodes: [l__mod___blocks_3_norm2, mul_17, x_72], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf161, buf162, buf163, buf164, buf165, 6272, 6, grid=grid(6272), stream=stream0)
        buf167 = reinterpret_tensor(buf159, (8, 784, 768), (602112, 768, 1), 0); del buf159  # reuse
        # Source Nodes: [l__mod___blocks_3_norm2, mul_17, x_72], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf153, arg14_1, buf160, arg185_1, buf164, buf165, arg186_1, arg187_1, buf167, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg186_1
        del arg187_1
        buf168 = reinterpret_tensor(buf132, (6272, 3072), (3072, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (6272, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 3072), (1, 768), 0), out=buf168)
        del arg188_1
        buf169 = reinterpret_tensor(buf168, (8, 784, 3072), (2408448, 3072, 1), 0); del buf168  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf169, arg189_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg189_1
        buf170 = reinterpret_tensor(buf167, (6272, 768), (768, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg190_1, (3072, 768), (1, 3072), 0), out=buf170)
        del arg190_1
        buf171 = reinterpret_tensor(buf170, (8, 784, 768), (602112, 768, 1), 0); del buf170  # reuse
        buf175 = reinterpret_tensor(buf158, (8, 784, 768), (602112, 768, 1), 0); del buf158  # reuse
        # Source Nodes: [l__mod___blocks_4_norm1, mul_17, mul_18, x_72, x_80], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf171, buf153, arg14_1, buf160, arg185_1, arg15_1, arg191_1, arg192_1, arg193_1, buf175, 6272, 768, grid=grid(6272), stream=stream0)
        del arg14_1
        del arg15_1
        del arg185_1
        del arg191_1
        del arg192_1
        del arg193_1
        del buf153
        buf176 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (6272, 768), (768, 1), 0), reinterpret_tensor(arg194_1, (768, 2304), (1, 768), 0), out=buf176)
        del arg194_1
        buf177 = buf141; del buf141  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf176, arg195_1, buf177, 43008, 112, grid=grid(43008), stream=stream0)
        buf178 = buf142; del buf142  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf177, buf178, 6144, 7, grid=grid(6144), stream=stream0)
        buf179 = buf177; del buf177  # reuse
        # Source Nodes: [k_9], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf176, arg195_1, buf179, 43008, 112, grid=grid(43008), stream=stream0)
        buf180 = buf140; del buf140  # reuse
        # Source Nodes: [k_9], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf179, buf180, 6144, 7, grid=grid(6144), stream=stream0)
        buf181 = reinterpret_tensor(buf175, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf175  # reuse
        # Source Nodes: [matmul_8, q_9], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf176, arg195_1, buf178, buf181, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf182 = reinterpret_tensor(buf160, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf160  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf176, arg195_1, buf180, buf182, 4816896, grid=grid(4816896), stream=stream0)
        buf183 = reinterpret_tensor(buf148, (128, 48, 48), (2304, 48, 1), 0); del buf148  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf181, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf182, (128, 784, 48), (37632, 48, 1), 0), out=buf183)
        buf186 = reinterpret_tensor(buf145, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf145  # reuse
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf183, arg17_1, buf186, 6144, 48, grid=grid(6144), stream=stream0)
        del arg17_1
        buf187 = reinterpret_tensor(buf182, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf182  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf176, arg195_1, buf187, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg195_1
        buf188 = reinterpret_tensor(buf181, (128, 48, 784), (37632, 784, 1), 0); del buf181  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf187, (128, 48, 784), (37632, 784, 1), 0), out=buf188)
        buf189 = reinterpret_tensor(buf187, (8, 784, 768), (602112, 768, 1), 0); del buf187  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf188, buf189, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf190 = reinterpret_tensor(buf188, (6272, 768), (768, 1), 0); del buf188  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (6272, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 768), (1, 768), 0), out=buf190)
        del arg196_1
        buf194 = buf189; del buf189  # reuse
        # Source Nodes: [l__mod___blocks_4_norm3, mul_20, x_82, x_84], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf171, arg16_1, buf190, arg197_1, arg198_1, arg199_1, buf194, 6272, 768, grid=grid(6272), stream=stream0)
        del arg198_1
        del arg199_1
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(reinterpret_tensor(buf194, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg200_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf195, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg200_1
        buf196 = reinterpret_tensor(buf194, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf194  # reuse
        # Source Nodes: [x_86, x_87, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf195, arg201_1, arg649_1, arg650_1, arg202_1, arg203_1, buf196, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg201_1
        del arg202_1
        del arg203_1
        del arg649_1
        del arg650_1
        # Source Nodes: [x_86, x_87, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf197 = extern_kernels.convolution(buf196, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf197, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg204_1
        buf198 = reinterpret_tensor(buf196, (8, 784, 768), (602112, 768, 1), 0); del buf196  # reuse
        buf202 = reinterpret_tensor(buf195, (8, 784, 768), (602112, 768, 1), 0); del buf195  # reuse
        # Source Nodes: [l__mod___blocks_4_norm2, mul_20, mul_21, x_82, x_84, x_91], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf171, arg16_1, buf190, arg197_1, arg18_1, buf197, arg205_1, arg206_1, arg207_1, buf198, buf202, 6272, 768, grid=grid(6272), stream=stream0)
        del arg16_1
        del arg18_1
        del arg197_1
        del arg205_1
        del arg206_1
        del arg207_1
        del buf171
        buf203 = reinterpret_tensor(buf169, (6272, 3072), (3072, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (6272, 768), (768, 1), 0), reinterpret_tensor(arg208_1, (768, 3072), (1, 768), 0), out=buf203)
        del arg208_1
        buf204 = reinterpret_tensor(buf203, (8, 784, 3072), (2408448, 3072, 1), 0); del buf203  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf204, arg209_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg209_1
        buf205 = reinterpret_tensor(buf202, (6272, 768), (768, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg210_1, (3072, 768), (1, 3072), 0), out=buf205)
        del arg210_1
        buf209 = reinterpret_tensor(buf197, (8, 784, 768), (602112, 768, 1), 0); del buf197  # reuse
        # Source Nodes: [l__mod___blocks_5_norm1, mul_22, x_99], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf198, arg19_1, buf205, arg211_1, arg212_1, arg213_1, buf209, 6272, 768, grid=grid(6272), stream=stream0)
        del arg212_1
        del arg213_1
        buf210 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (6272, 768), (768, 1), 0), reinterpret_tensor(arg214_1, (768, 2304), (1, 768), 0), out=buf210)
        del arg214_1
        buf211 = buf179; del buf179  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf210, arg215_1, buf211, 43008, 112, grid=grid(43008), stream=stream0)
        buf212 = buf180; del buf180  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf211, buf212, 6144, 7, grid=grid(6144), stream=stream0)
        buf213 = buf211; del buf211  # reuse
        # Source Nodes: [k_11], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf210, arg215_1, buf213, 43008, 112, grid=grid(43008), stream=stream0)
        buf214 = buf178; del buf178  # reuse
        # Source Nodes: [k_11], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf213, buf214, 6144, 7, grid=grid(6144), stream=stream0)
        buf215 = reinterpret_tensor(buf209, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf209  # reuse
        # Source Nodes: [matmul_10, q_11], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf210, arg215_1, buf212, buf215, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf216 = reinterpret_tensor(buf190, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf190  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf210, arg215_1, buf214, buf216, 4816896, grid=grid(4816896), stream=stream0)
        buf217 = reinterpret_tensor(buf186, (128, 48, 48), (2304, 48, 1), 0); del buf186  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf216, (128, 784, 48), (37632, 48, 1), 0), out=buf217)
        buf220 = reinterpret_tensor(buf183, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf183  # reuse
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf217, arg21_1, buf220, 6144, 48, grid=grid(6144), stream=stream0)
        del arg21_1
        buf221 = reinterpret_tensor(buf216, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf216  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf210, arg215_1, buf221, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg215_1
        buf222 = reinterpret_tensor(buf215, (128, 48, 784), (37632, 784, 1), 0); del buf215  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf221, (128, 48, 784), (37632, 784, 1), 0), out=buf222)
        buf223 = reinterpret_tensor(buf221, (8, 784, 768), (602112, 768, 1), 0); del buf221  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf222, buf223, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf224 = reinterpret_tensor(buf222, (6272, 768), (768, 1), 0); del buf222  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (6272, 768), (768, 1), 0), reinterpret_tensor(arg216_1, (768, 768), (1, 768), 0), out=buf224)
        del arg216_1
        buf225 = reinterpret_tensor(buf224, (8, 784, 768), (602112, 768, 1), 0); del buf224  # reuse
        buf229 = buf223; del buf223  # reuse
        # Source Nodes: [l__mod___blocks_5_norm3, mul_22, mul_24, x_101, x_103, x_99], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf225, buf198, arg19_1, buf205, arg211_1, arg20_1, arg217_1, arg218_1, arg219_1, buf229, 6272, 768, grid=grid(6272), stream=stream0)
        del arg19_1
        del arg20_1
        del arg211_1
        del arg217_1
        del arg218_1
        del arg219_1
        del buf198
        del buf205
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(reinterpret_tensor(buf229, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg220_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf230, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg220_1
        buf231 = reinterpret_tensor(buf229, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf229  # reuse
        # Source Nodes: [x_105, x_106, x_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf230, arg221_1, arg652_1, arg653_1, arg222_1, arg223_1, buf231, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del arg652_1
        del arg653_1
        # Source Nodes: [x_105, x_106, x_107, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf232 = extern_kernels.convolution(buf231, arg224_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf232, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg224_1
        buf233 = buf163; del buf163  # reuse
        buf234 = buf162; del buf162  # reuse
        buf235 = buf161; del buf161  # reuse
        # Source Nodes: [l__mod___blocks_5_norm2, mul_25, x_110], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf225, arg22_1, buf232, arg225_1, buf233, buf234, buf235, 37632, 128, grid=grid(37632), stream=stream0)
        buf236 = buf165; del buf165  # reuse
        buf237 = buf164; del buf164  # reuse
        # Source Nodes: [l__mod___blocks_5_norm2, mul_25, x_110], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf233, buf234, buf235, buf236, buf237, 6272, 6, grid=grid(6272), stream=stream0)
        buf239 = reinterpret_tensor(buf231, (8, 784, 768), (602112, 768, 1), 0); del buf231  # reuse
        # Source Nodes: [l__mod___blocks_5_norm2, mul_25, x_110], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf225, arg22_1, buf232, arg225_1, buf236, buf237, arg226_1, arg227_1, buf239, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg226_1
        del arg227_1
        buf240 = reinterpret_tensor(buf204, (6272, 3072), (3072, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (6272, 768), (768, 1), 0), reinterpret_tensor(arg228_1, (768, 3072), (1, 768), 0), out=buf240)
        del arg228_1
        buf241 = reinterpret_tensor(buf240, (8, 784, 3072), (2408448, 3072, 1), 0); del buf240  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf241, arg229_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg229_1
        buf242 = reinterpret_tensor(buf239, (6272, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg230_1, (3072, 768), (1, 3072), 0), out=buf242)
        del arg230_1
        buf243 = reinterpret_tensor(buf242, (8, 784, 768), (602112, 768, 1), 0); del buf242  # reuse
        buf247 = reinterpret_tensor(buf230, (8, 784, 768), (602112, 768, 1), 0); del buf230  # reuse
        # Source Nodes: [l__mod___blocks_6_norm1, mul_25, mul_26, x_110, x_118], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf243, buf225, arg22_1, buf232, arg225_1, arg23_1, arg231_1, arg232_1, arg233_1, buf247, 6272, 768, grid=grid(6272), stream=stream0)
        del arg225_1
        del arg22_1
        del arg231_1
        del arg232_1
        del arg233_1
        del arg23_1
        del buf225
        buf248 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (6272, 768), (768, 1), 0), reinterpret_tensor(arg234_1, (768, 2304), (1, 768), 0), out=buf248)
        del arg234_1
        buf249 = buf213; del buf213  # reuse
        # Source Nodes: [q_13], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf248, arg235_1, buf249, 43008, 112, grid=grid(43008), stream=stream0)
        buf250 = buf214; del buf214  # reuse
        # Source Nodes: [q_13], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf249, buf250, 6144, 7, grid=grid(6144), stream=stream0)
        buf251 = buf249; del buf249  # reuse
        # Source Nodes: [k_13], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf248, arg235_1, buf251, 43008, 112, grid=grid(43008), stream=stream0)
        buf252 = buf212; del buf212  # reuse
        # Source Nodes: [k_13], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf251, buf252, 6144, 7, grid=grid(6144), stream=stream0)
        buf253 = reinterpret_tensor(buf247, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf247  # reuse
        # Source Nodes: [matmul_12, q_13], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf248, arg235_1, buf250, buf253, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf254 = reinterpret_tensor(buf232, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf232  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf248, arg235_1, buf252, buf254, 4816896, grid=grid(4816896), stream=stream0)
        buf255 = reinterpret_tensor(buf220, (128, 48, 48), (2304, 48, 1), 0); del buf220  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf253, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf254, (128, 784, 48), (37632, 48, 1), 0), out=buf255)
        buf258 = reinterpret_tensor(buf217, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf217  # reuse
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf255, arg25_1, buf258, 6144, 48, grid=grid(6144), stream=stream0)
        del arg25_1
        buf259 = reinterpret_tensor(buf254, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf254  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf248, arg235_1, buf259, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg235_1
        buf260 = reinterpret_tensor(buf253, (128, 48, 784), (37632, 784, 1), 0); del buf253  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf259, (128, 48, 784), (37632, 784, 1), 0), out=buf260)
        buf261 = reinterpret_tensor(buf259, (8, 784, 768), (602112, 768, 1), 0); del buf259  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf260, buf261, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf262 = reinterpret_tensor(buf260, (6272, 768), (768, 1), 0); del buf260  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (6272, 768), (768, 1), 0), reinterpret_tensor(arg236_1, (768, 768), (1, 768), 0), out=buf262)
        del arg236_1
        buf266 = buf261; del buf261  # reuse
        # Source Nodes: [l__mod___blocks_6_norm3, mul_28, x_120, x_122], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf243, arg24_1, buf262, arg237_1, arg238_1, arg239_1, buf266, 6272, 768, grid=grid(6272), stream=stream0)
        del arg238_1
        del arg239_1
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(reinterpret_tensor(buf266, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg240_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf267, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg240_1
        buf268 = reinterpret_tensor(buf266, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf266  # reuse
        # Source Nodes: [x_124, x_125, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf267, arg241_1, arg655_1, arg656_1, arg242_1, arg243_1, buf268, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        del arg655_1
        del arg656_1
        # Source Nodes: [x_124, x_125, x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf269 = extern_kernels.convolution(buf268, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf269, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg244_1
        buf270 = reinterpret_tensor(buf268, (8, 784, 768), (602112, 768, 1), 0); del buf268  # reuse
        buf274 = reinterpret_tensor(buf267, (8, 784, 768), (602112, 768, 1), 0); del buf267  # reuse
        # Source Nodes: [l__mod___blocks_6_norm2, mul_28, mul_29, x_120, x_122, x_129], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf243, arg24_1, buf262, arg237_1, arg26_1, buf269, arg245_1, arg246_1, arg247_1, buf270, buf274, 6272, 768, grid=grid(6272), stream=stream0)
        del arg237_1
        del arg245_1
        del arg246_1
        del arg247_1
        del arg24_1
        del arg26_1
        del buf243
        buf275 = reinterpret_tensor(buf241, (6272, 3072), (3072, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (6272, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 3072), (1, 768), 0), out=buf275)
        del arg248_1
        buf276 = reinterpret_tensor(buf275, (8, 784, 3072), (2408448, 3072, 1), 0); del buf275  # reuse
        # Source Nodes: [x_131], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf276, arg249_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg249_1
        buf277 = reinterpret_tensor(buf274, (6272, 768), (768, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg250_1, (3072, 768), (1, 3072), 0), out=buf277)
        del arg250_1
        buf281 = reinterpret_tensor(buf269, (8, 784, 768), (602112, 768, 1), 0); del buf269  # reuse
        # Source Nodes: [l__mod___blocks_7_norm1, mul_30, x_137], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf270, arg27_1, buf277, arg251_1, arg252_1, arg253_1, buf281, 6272, 768, grid=grid(6272), stream=stream0)
        del arg252_1
        del arg253_1
        buf282 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (6272, 768), (768, 1), 0), reinterpret_tensor(arg254_1, (768, 2304), (1, 768), 0), out=buf282)
        del arg254_1
        buf283 = buf251; del buf251  # reuse
        # Source Nodes: [q_15], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf282, arg255_1, buf283, 43008, 112, grid=grid(43008), stream=stream0)
        buf284 = buf252; del buf252  # reuse
        # Source Nodes: [q_15], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf283, buf284, 6144, 7, grid=grid(6144), stream=stream0)
        buf285 = buf283; del buf283  # reuse
        # Source Nodes: [k_15], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf282, arg255_1, buf285, 43008, 112, grid=grid(43008), stream=stream0)
        buf286 = buf250; del buf250  # reuse
        # Source Nodes: [k_15], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf285, buf286, 6144, 7, grid=grid(6144), stream=stream0)
        buf287 = reinterpret_tensor(buf281, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf281  # reuse
        # Source Nodes: [matmul_14, q_15], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf282, arg255_1, buf284, buf287, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf288 = reinterpret_tensor(buf262, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf262  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf282, arg255_1, buf286, buf288, 4816896, grid=grid(4816896), stream=stream0)
        buf289 = reinterpret_tensor(buf258, (128, 48, 48), (2304, 48, 1), 0); del buf258  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf287, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf288, (128, 784, 48), (37632, 48, 1), 0), out=buf289)
        buf292 = reinterpret_tensor(buf255, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf255  # reuse
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf289, arg29_1, buf292, 6144, 48, grid=grid(6144), stream=stream0)
        del arg29_1
        buf293 = reinterpret_tensor(buf288, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf288  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf282, arg255_1, buf293, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg255_1
        buf294 = reinterpret_tensor(buf287, (128, 48, 784), (37632, 784, 1), 0); del buf287  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf293, (128, 48, 784), (37632, 784, 1), 0), out=buf294)
        buf295 = reinterpret_tensor(buf293, (8, 784, 768), (602112, 768, 1), 0); del buf293  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf294, buf295, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf296 = reinterpret_tensor(buf294, (6272, 768), (768, 1), 0); del buf294  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (6272, 768), (768, 1), 0), reinterpret_tensor(arg256_1, (768, 768), (1, 768), 0), out=buf296)
        del arg256_1
        buf297 = reinterpret_tensor(buf296, (8, 784, 768), (602112, 768, 1), 0); del buf296  # reuse
        buf301 = buf295; del buf295  # reuse
        # Source Nodes: [l__mod___blocks_7_norm3, mul_30, mul_32, x_137, x_139, x_141], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf297, buf270, arg27_1, buf277, arg251_1, arg28_1, arg257_1, arg258_1, arg259_1, buf301, 6272, 768, grid=grid(6272), stream=stream0)
        del arg251_1
        del arg257_1
        del arg258_1
        del arg259_1
        del arg27_1
        del arg28_1
        del buf270
        del buf277
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(reinterpret_tensor(buf301, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg260_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf302, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg260_1
        buf303 = reinterpret_tensor(buf301, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf301  # reuse
        # Source Nodes: [x_143, x_144, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf302, arg261_1, arg658_1, arg659_1, arg262_1, arg263_1, buf303, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg261_1
        del arg262_1
        del arg263_1
        del arg658_1
        del arg659_1
        # Source Nodes: [x_143, x_144, x_145, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf304 = extern_kernels.convolution(buf303, arg264_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf304, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg264_1
        buf305 = buf235; del buf235  # reuse
        buf306 = buf234; del buf234  # reuse
        buf307 = buf233; del buf233  # reuse
        # Source Nodes: [l__mod___blocks_7_norm2, mul_33, x_148], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf297, arg30_1, buf304, arg265_1, buf305, buf306, buf307, 37632, 128, grid=grid(37632), stream=stream0)
        buf308 = buf237; del buf237  # reuse
        buf309 = buf236; del buf236  # reuse
        # Source Nodes: [l__mod___blocks_7_norm2, mul_33, x_148], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf305, buf306, buf307, buf308, buf309, 6272, 6, grid=grid(6272), stream=stream0)
        buf311 = reinterpret_tensor(buf303, (8, 784, 768), (602112, 768, 1), 0); del buf303  # reuse
        # Source Nodes: [l__mod___blocks_7_norm2, mul_33, x_148], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf297, arg30_1, buf304, arg265_1, buf308, buf309, arg266_1, arg267_1, buf311, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg266_1
        del arg267_1
        buf312 = reinterpret_tensor(buf276, (6272, 3072), (3072, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (6272, 768), (768, 1), 0), reinterpret_tensor(arg268_1, (768, 3072), (1, 768), 0), out=buf312)
        del arg268_1
        buf313 = reinterpret_tensor(buf312, (8, 784, 3072), (2408448, 3072, 1), 0); del buf312  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf313, arg269_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg269_1
        buf314 = reinterpret_tensor(buf311, (6272, 768), (768, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg270_1, (3072, 768), (1, 3072), 0), out=buf314)
        del arg270_1
        buf315 = reinterpret_tensor(buf314, (8, 784, 768), (602112, 768, 1), 0); del buf314  # reuse
        buf319 = reinterpret_tensor(buf302, (8, 784, 768), (602112, 768, 1), 0); del buf302  # reuse
        # Source Nodes: [l__mod___blocks_8_norm1, mul_33, mul_34, x_148, x_156], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf315, buf297, arg30_1, buf304, arg265_1, arg31_1, arg271_1, arg272_1, arg273_1, buf319, 6272, 768, grid=grid(6272), stream=stream0)
        del arg265_1
        del arg271_1
        del arg272_1
        del arg273_1
        del arg30_1
        del arg31_1
        del buf297
        buf320 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (6272, 768), (768, 1), 0), reinterpret_tensor(arg274_1, (768, 2304), (1, 768), 0), out=buf320)
        del arg274_1
        buf321 = buf285; del buf285  # reuse
        # Source Nodes: [q_17], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf320, arg275_1, buf321, 43008, 112, grid=grid(43008), stream=stream0)
        buf322 = buf286; del buf286  # reuse
        # Source Nodes: [q_17], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf321, buf322, 6144, 7, grid=grid(6144), stream=stream0)
        buf323 = buf321; del buf321  # reuse
        # Source Nodes: [k_17], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf320, arg275_1, buf323, 43008, 112, grid=grid(43008), stream=stream0)
        buf324 = buf284; del buf284  # reuse
        # Source Nodes: [k_17], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf323, buf324, 6144, 7, grid=grid(6144), stream=stream0)
        buf325 = reinterpret_tensor(buf319, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf319  # reuse
        # Source Nodes: [matmul_16, q_17], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf320, arg275_1, buf322, buf325, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf326 = reinterpret_tensor(buf304, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf304  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf320, arg275_1, buf324, buf326, 4816896, grid=grid(4816896), stream=stream0)
        buf327 = reinterpret_tensor(buf292, (128, 48, 48), (2304, 48, 1), 0); del buf292  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf326, (128, 784, 48), (37632, 48, 1), 0), out=buf327)
        buf330 = reinterpret_tensor(buf289, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf289  # reuse
        # Source Nodes: [attn_24, attn_25], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf327, arg33_1, buf330, 6144, 48, grid=grid(6144), stream=stream0)
        del arg33_1
        buf331 = reinterpret_tensor(buf326, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf326  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf320, arg275_1, buf331, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg275_1
        buf332 = reinterpret_tensor(buf325, (128, 48, 784), (37632, 784, 1), 0); del buf325  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf331, (128, 48, 784), (37632, 784, 1), 0), out=buf332)
        buf333 = reinterpret_tensor(buf331, (8, 784, 768), (602112, 768, 1), 0); del buf331  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf332, buf333, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf334 = reinterpret_tensor(buf332, (6272, 768), (768, 1), 0); del buf332  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (6272, 768), (768, 1), 0), reinterpret_tensor(arg276_1, (768, 768), (1, 768), 0), out=buf334)
        del arg276_1
        buf338 = buf333; del buf333  # reuse
        # Source Nodes: [l__mod___blocks_8_norm3, mul_36, x_158, x_160], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf315, arg32_1, buf334, arg277_1, arg278_1, arg279_1, buf338, 6272, 768, grid=grid(6272), stream=stream0)
        del arg278_1
        del arg279_1
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(reinterpret_tensor(buf338, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg280_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf339, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg280_1
        buf340 = reinterpret_tensor(buf338, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf338  # reuse
        # Source Nodes: [x_162, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf339, arg281_1, arg661_1, arg662_1, arg282_1, arg283_1, buf340, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg281_1
        del arg282_1
        del arg283_1
        del arg661_1
        del arg662_1
        # Source Nodes: [x_162, x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf341 = extern_kernels.convolution(buf340, arg284_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf341, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg284_1
        buf342 = reinterpret_tensor(buf340, (8, 784, 768), (602112, 768, 1), 0); del buf340  # reuse
        buf346 = reinterpret_tensor(buf339, (8, 784, 768), (602112, 768, 1), 0); del buf339  # reuse
        # Source Nodes: [l__mod___blocks_8_norm2, mul_36, mul_37, x_158, x_160, x_167], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf315, arg32_1, buf334, arg277_1, arg34_1, buf341, arg285_1, arg286_1, arg287_1, buf342, buf346, 6272, 768, grid=grid(6272), stream=stream0)
        del arg277_1
        del arg285_1
        del arg286_1
        del arg287_1
        del arg32_1
        del arg34_1
        del buf315
        buf347 = reinterpret_tensor(buf313, (6272, 3072), (3072, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (6272, 768), (768, 1), 0), reinterpret_tensor(arg288_1, (768, 3072), (1, 768), 0), out=buf347)
        del arg288_1
        buf348 = reinterpret_tensor(buf347, (8, 784, 3072), (2408448, 3072, 1), 0); del buf347  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf348, arg289_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg289_1
        buf349 = reinterpret_tensor(buf346, (6272, 768), (768, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg290_1, (3072, 768), (1, 3072), 0), out=buf349)
        del arg290_1
        buf353 = reinterpret_tensor(buf341, (8, 784, 768), (602112, 768, 1), 0); del buf341  # reuse
        # Source Nodes: [l__mod___blocks_9_norm1, mul_38, x_175], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf342, arg35_1, buf349, arg291_1, arg292_1, arg293_1, buf353, 6272, 768, grid=grid(6272), stream=stream0)
        del arg292_1
        del arg293_1
        buf354 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (6272, 768), (768, 1), 0), reinterpret_tensor(arg294_1, (768, 2304), (1, 768), 0), out=buf354)
        del arg294_1
        buf355 = buf323; del buf323  # reuse
        # Source Nodes: [q_19], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf354, arg295_1, buf355, 43008, 112, grid=grid(43008), stream=stream0)
        buf356 = buf324; del buf324  # reuse
        # Source Nodes: [q_19], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf355, buf356, 6144, 7, grid=grid(6144), stream=stream0)
        buf357 = buf355; del buf355  # reuse
        # Source Nodes: [k_19], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf354, arg295_1, buf357, 43008, 112, grid=grid(43008), stream=stream0)
        buf358 = buf322; del buf322  # reuse
        # Source Nodes: [k_19], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf357, buf358, 6144, 7, grid=grid(6144), stream=stream0)
        buf359 = reinterpret_tensor(buf353, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf353  # reuse
        # Source Nodes: [matmul_18, q_19], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf354, arg295_1, buf356, buf359, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf360 = reinterpret_tensor(buf334, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf334  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf354, arg295_1, buf358, buf360, 4816896, grid=grid(4816896), stream=stream0)
        buf361 = reinterpret_tensor(buf330, (128, 48, 48), (2304, 48, 1), 0); del buf330  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf359, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf360, (128, 784, 48), (37632, 48, 1), 0), out=buf361)
        buf364 = reinterpret_tensor(buf327, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf327  # reuse
        # Source Nodes: [attn_27, attn_28], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf361, arg37_1, buf364, 6144, 48, grid=grid(6144), stream=stream0)
        del arg37_1
        buf365 = reinterpret_tensor(buf360, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf360  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf354, arg295_1, buf365, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg295_1
        buf366 = reinterpret_tensor(buf359, (128, 48, 784), (37632, 784, 1), 0); del buf359  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf365, (128, 48, 784), (37632, 784, 1), 0), out=buf366)
        buf367 = reinterpret_tensor(buf365, (8, 784, 768), (602112, 768, 1), 0); del buf365  # reuse
        # Source Nodes: [x_177], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf366, buf367, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf368 = reinterpret_tensor(buf366, (6272, 768), (768, 1), 0); del buf366  # reuse
        # Source Nodes: [x_177], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (6272, 768), (768, 1), 0), reinterpret_tensor(arg296_1, (768, 768), (1, 768), 0), out=buf368)
        del arg296_1
        buf369 = reinterpret_tensor(buf368, (8, 784, 768), (602112, 768, 1), 0); del buf368  # reuse
        buf373 = buf367; del buf367  # reuse
        # Source Nodes: [l__mod___blocks_9_norm3, mul_38, mul_40, x_175, x_177, x_179], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf369, buf342, arg35_1, buf349, arg291_1, arg36_1, arg297_1, arg298_1, arg299_1, buf373, 6272, 768, grid=grid(6272), stream=stream0)
        del arg291_1
        del arg297_1
        del arg298_1
        del arg299_1
        del arg35_1
        del arg36_1
        del buf342
        del buf349
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(reinterpret_tensor(buf373, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg300_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf374, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg300_1
        buf375 = reinterpret_tensor(buf373, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf373  # reuse
        # Source Nodes: [x_181, x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf374, arg301_1, arg664_1, arg665_1, arg302_1, arg303_1, buf375, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        del arg664_1
        del arg665_1
        # Source Nodes: [x_181, x_182, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf376 = extern_kernels.convolution(buf375, arg304_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf376, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg304_1
        buf377 = buf307; del buf307  # reuse
        buf378 = buf306; del buf306  # reuse
        buf379 = buf305; del buf305  # reuse
        # Source Nodes: [l__mod___blocks_9_norm2, mul_41, x_186], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf369, arg38_1, buf376, arg305_1, buf377, buf378, buf379, 37632, 128, grid=grid(37632), stream=stream0)
        buf380 = buf309; del buf309  # reuse
        buf381 = buf308; del buf308  # reuse
        # Source Nodes: [l__mod___blocks_9_norm2, mul_41, x_186], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf377, buf378, buf379, buf380, buf381, 6272, 6, grid=grid(6272), stream=stream0)
        buf383 = reinterpret_tensor(buf375, (8, 784, 768), (602112, 768, 1), 0); del buf375  # reuse
        # Source Nodes: [l__mod___blocks_9_norm2, mul_41, x_186], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf369, arg38_1, buf376, arg305_1, buf380, buf381, arg306_1, arg307_1, buf383, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg306_1
        del arg307_1
        buf384 = reinterpret_tensor(buf348, (6272, 3072), (3072, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (6272, 768), (768, 1), 0), reinterpret_tensor(arg308_1, (768, 3072), (1, 768), 0), out=buf384)
        del arg308_1
        buf385 = reinterpret_tensor(buf384, (8, 784, 3072), (2408448, 3072, 1), 0); del buf384  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf385, arg309_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg309_1
        buf386 = reinterpret_tensor(buf383, (6272, 768), (768, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg310_1, (3072, 768), (1, 3072), 0), out=buf386)
        del arg310_1
        buf387 = reinterpret_tensor(buf386, (8, 784, 768), (602112, 768, 1), 0); del buf386  # reuse
        buf391 = reinterpret_tensor(buf374, (8, 784, 768), (602112, 768, 1), 0); del buf374  # reuse
        # Source Nodes: [l__mod___blocks_10_norm1, mul_41, mul_42, x_186, x_194], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf387, buf369, arg38_1, buf376, arg305_1, arg39_1, arg311_1, arg312_1, arg313_1, buf391, 6272, 768, grid=grid(6272), stream=stream0)
        del arg305_1
        del arg311_1
        del arg312_1
        del arg313_1
        del arg38_1
        del arg39_1
        del buf369
        buf392 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (6272, 768), (768, 1), 0), reinterpret_tensor(arg314_1, (768, 2304), (1, 768), 0), out=buf392)
        del arg314_1
        buf393 = buf357; del buf357  # reuse
        # Source Nodes: [q_21], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf392, arg315_1, buf393, 43008, 112, grid=grid(43008), stream=stream0)
        buf394 = buf358; del buf358  # reuse
        # Source Nodes: [q_21], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf393, buf394, 6144, 7, grid=grid(6144), stream=stream0)
        buf395 = buf393; del buf393  # reuse
        # Source Nodes: [k_21], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf392, arg315_1, buf395, 43008, 112, grid=grid(43008), stream=stream0)
        buf396 = buf356; del buf356  # reuse
        # Source Nodes: [k_21], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf395, buf396, 6144, 7, grid=grid(6144), stream=stream0)
        buf397 = reinterpret_tensor(buf391, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf391  # reuse
        # Source Nodes: [matmul_20, q_21], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf392, arg315_1, buf394, buf397, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf398 = reinterpret_tensor(buf376, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf376  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf392, arg315_1, buf396, buf398, 4816896, grid=grid(4816896), stream=stream0)
        buf399 = reinterpret_tensor(buf364, (128, 48, 48), (2304, 48, 1), 0); del buf364  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf398, (128, 784, 48), (37632, 48, 1), 0), out=buf399)
        buf402 = reinterpret_tensor(buf361, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf361  # reuse
        # Source Nodes: [attn_30, attn_31], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf399, arg41_1, buf402, 6144, 48, grid=grid(6144), stream=stream0)
        del arg41_1
        buf403 = reinterpret_tensor(buf398, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf398  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf392, arg315_1, buf403, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg315_1
        buf404 = reinterpret_tensor(buf397, (128, 48, 784), (37632, 784, 1), 0); del buf397  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf403, (128, 48, 784), (37632, 784, 1), 0), out=buf404)
        buf405 = reinterpret_tensor(buf403, (8, 784, 768), (602112, 768, 1), 0); del buf403  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf404, buf405, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (6272, 768), (768, 1), 0); del buf404  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (6272, 768), (768, 1), 0), reinterpret_tensor(arg316_1, (768, 768), (1, 768), 0), out=buf406)
        del arg316_1
        buf410 = buf405; del buf405  # reuse
        # Source Nodes: [l__mod___blocks_10_norm3, mul_44, x_196, x_198], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf387, arg40_1, buf406, arg317_1, arg318_1, arg319_1, buf410, 6272, 768, grid=grid(6272), stream=stream0)
        del arg318_1
        del arg319_1
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(reinterpret_tensor(buf410, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg320_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf411, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg320_1
        buf412 = reinterpret_tensor(buf410, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf410  # reuse
        # Source Nodes: [x_200, x_201, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf411, arg321_1, arg667_1, arg668_1, arg322_1, arg323_1, buf412, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg321_1
        del arg322_1
        del arg323_1
        del arg667_1
        del arg668_1
        # Source Nodes: [x_200, x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf413 = extern_kernels.convolution(buf412, arg324_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf413, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg324_1
        buf414 = reinterpret_tensor(buf412, (8, 784, 768), (602112, 768, 1), 0); del buf412  # reuse
        buf418 = reinterpret_tensor(buf411, (8, 784, 768), (602112, 768, 1), 0); del buf411  # reuse
        # Source Nodes: [l__mod___blocks_10_norm2, mul_44, mul_45, x_196, x_198, x_205], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf387, arg40_1, buf406, arg317_1, arg42_1, buf413, arg325_1, arg326_1, arg327_1, buf414, buf418, 6272, 768, grid=grid(6272), stream=stream0)
        del arg317_1
        del arg325_1
        del arg326_1
        del arg327_1
        del arg40_1
        del arg42_1
        del buf387
        buf419 = reinterpret_tensor(buf385, (6272, 3072), (3072, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (6272, 768), (768, 1), 0), reinterpret_tensor(arg328_1, (768, 3072), (1, 768), 0), out=buf419)
        del arg328_1
        buf420 = reinterpret_tensor(buf419, (8, 784, 3072), (2408448, 3072, 1), 0); del buf419  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf420, arg329_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg329_1
        buf421 = reinterpret_tensor(buf418, (6272, 768), (768, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf420, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg330_1, (3072, 768), (1, 3072), 0), out=buf421)
        del arg330_1
        buf425 = reinterpret_tensor(buf413, (8, 784, 768), (602112, 768, 1), 0); del buf413  # reuse
        # Source Nodes: [l__mod___blocks_11_norm1, mul_46, x_213], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf414, arg43_1, buf421, arg331_1, arg332_1, arg333_1, buf425, 6272, 768, grid=grid(6272), stream=stream0)
        del arg332_1
        del arg333_1
        buf426 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf425, (6272, 768), (768, 1), 0), reinterpret_tensor(arg334_1, (768, 2304), (1, 768), 0), out=buf426)
        del arg334_1
        buf427 = buf395; del buf395  # reuse
        # Source Nodes: [q_23], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf426, arg335_1, buf427, 43008, 112, grid=grid(43008), stream=stream0)
        buf428 = buf396; del buf396  # reuse
        # Source Nodes: [q_23], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf427, buf428, 6144, 7, grid=grid(6144), stream=stream0)
        buf429 = buf427; del buf427  # reuse
        # Source Nodes: [k_23], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf426, arg335_1, buf429, 43008, 112, grid=grid(43008), stream=stream0)
        buf430 = buf394; del buf394  # reuse
        # Source Nodes: [k_23], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf429, buf430, 6144, 7, grid=grid(6144), stream=stream0)
        buf431 = reinterpret_tensor(buf425, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf425  # reuse
        # Source Nodes: [matmul_22, q_23], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf426, arg335_1, buf428, buf431, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf432 = reinterpret_tensor(buf406, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf406  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf426, arg335_1, buf430, buf432, 4816896, grid=grid(4816896), stream=stream0)
        buf433 = reinterpret_tensor(buf402, (128, 48, 48), (2304, 48, 1), 0); del buf402  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf431, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf432, (128, 784, 48), (37632, 48, 1), 0), out=buf433)
        buf436 = reinterpret_tensor(buf399, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf399  # reuse
        # Source Nodes: [attn_33, attn_34], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf433, arg45_1, buf436, 6144, 48, grid=grid(6144), stream=stream0)
        del arg45_1
        buf437 = reinterpret_tensor(buf432, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf432  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf426, arg335_1, buf437, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg335_1
        buf438 = reinterpret_tensor(buf431, (128, 48, 784), (37632, 784, 1), 0); del buf431  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf437, (128, 48, 784), (37632, 784, 1), 0), out=buf438)
        buf439 = reinterpret_tensor(buf437, (8, 784, 768), (602112, 768, 1), 0); del buf437  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf438, buf439, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf440 = reinterpret_tensor(buf438, (6272, 768), (768, 1), 0); del buf438  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (6272, 768), (768, 1), 0), reinterpret_tensor(arg336_1, (768, 768), (1, 768), 0), out=buf440)
        del arg336_1
        buf441 = reinterpret_tensor(buf440, (8, 784, 768), (602112, 768, 1), 0); del buf440  # reuse
        buf445 = buf439; del buf439  # reuse
        # Source Nodes: [l__mod___blocks_11_norm3, mul_46, mul_48, x_213, x_215, x_217], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf441, buf414, arg43_1, buf421, arg331_1, arg44_1, arg337_1, arg338_1, arg339_1, buf445, 6272, 768, grid=grid(6272), stream=stream0)
        del arg331_1
        del arg337_1
        del arg338_1
        del arg339_1
        del arg43_1
        del arg44_1
        del buf414
        del buf421
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(reinterpret_tensor(buf445, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg340_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf446, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg340_1
        buf447 = reinterpret_tensor(buf445, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf445  # reuse
        # Source Nodes: [x_219, x_220, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf446, arg341_1, arg670_1, arg671_1, arg342_1, arg343_1, buf447, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg341_1
        del arg342_1
        del arg343_1
        del arg670_1
        del arg671_1
        # Source Nodes: [x_219, x_220, x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf448 = extern_kernels.convolution(buf447, arg344_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf448, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg344_1
        buf449 = buf379; del buf379  # reuse
        buf450 = buf378; del buf378  # reuse
        buf451 = buf377; del buf377  # reuse
        # Source Nodes: [l__mod___blocks_11_norm2, mul_49, x_224], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf441, arg46_1, buf448, arg345_1, buf449, buf450, buf451, 37632, 128, grid=grid(37632), stream=stream0)
        buf452 = buf381; del buf381  # reuse
        buf453 = buf380; del buf380  # reuse
        # Source Nodes: [l__mod___blocks_11_norm2, mul_49, x_224], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf449, buf450, buf451, buf452, buf453, 6272, 6, grid=grid(6272), stream=stream0)
        buf455 = reinterpret_tensor(buf447, (8, 784, 768), (602112, 768, 1), 0); del buf447  # reuse
        # Source Nodes: [l__mod___blocks_11_norm2, mul_49, x_224], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf441, arg46_1, buf448, arg345_1, buf452, buf453, arg346_1, arg347_1, buf455, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg346_1
        del arg347_1
        buf456 = reinterpret_tensor(buf420, (6272, 3072), (3072, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf455, (6272, 768), (768, 1), 0), reinterpret_tensor(arg348_1, (768, 3072), (1, 768), 0), out=buf456)
        del arg348_1
        buf457 = reinterpret_tensor(buf456, (8, 784, 3072), (2408448, 3072, 1), 0); del buf456  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf457, arg349_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg349_1
        buf458 = reinterpret_tensor(buf455, (6272, 768), (768, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg350_1, (3072, 768), (1, 3072), 0), out=buf458)
        del arg350_1
        buf459 = reinterpret_tensor(buf458, (8, 784, 768), (602112, 768, 1), 0); del buf458  # reuse
        buf463 = reinterpret_tensor(buf446, (8, 784, 768), (602112, 768, 1), 0); del buf446  # reuse
        # Source Nodes: [l__mod___blocks_12_norm1, mul_49, mul_50, x_224, x_232], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf459, buf441, arg46_1, buf448, arg345_1, arg47_1, arg351_1, arg352_1, arg353_1, buf463, 6272, 768, grid=grid(6272), stream=stream0)
        del arg345_1
        del arg351_1
        del arg352_1
        del arg353_1
        del arg46_1
        del arg47_1
        del buf441
        buf464 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (6272, 768), (768, 1), 0), reinterpret_tensor(arg354_1, (768, 2304), (1, 768), 0), out=buf464)
        del arg354_1
        buf465 = buf429; del buf429  # reuse
        # Source Nodes: [q_25], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf464, arg355_1, buf465, 43008, 112, grid=grid(43008), stream=stream0)
        buf466 = buf430; del buf430  # reuse
        # Source Nodes: [q_25], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf465, buf466, 6144, 7, grid=grid(6144), stream=stream0)
        buf467 = buf465; del buf465  # reuse
        # Source Nodes: [k_25], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf464, arg355_1, buf467, 43008, 112, grid=grid(43008), stream=stream0)
        buf468 = buf428; del buf428  # reuse
        # Source Nodes: [k_25], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf467, buf468, 6144, 7, grid=grid(6144), stream=stream0)
        buf469 = reinterpret_tensor(buf463, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf463  # reuse
        # Source Nodes: [matmul_24, q_25], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf464, arg355_1, buf466, buf469, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf470 = reinterpret_tensor(buf448, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf448  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf464, arg355_1, buf468, buf470, 4816896, grid=grid(4816896), stream=stream0)
        buf471 = reinterpret_tensor(buf436, (128, 48, 48), (2304, 48, 1), 0); del buf436  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf469, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf470, (128, 784, 48), (37632, 48, 1), 0), out=buf471)
        buf474 = reinterpret_tensor(buf433, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf433  # reuse
        # Source Nodes: [attn_36, attn_37], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf471, arg49_1, buf474, 6144, 48, grid=grid(6144), stream=stream0)
        del arg49_1
        buf475 = reinterpret_tensor(buf470, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf470  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf464, arg355_1, buf475, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg355_1
        buf476 = reinterpret_tensor(buf469, (128, 48, 784), (37632, 784, 1), 0); del buf469  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf474, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf475, (128, 48, 784), (37632, 784, 1), 0), out=buf476)
        buf477 = reinterpret_tensor(buf475, (8, 784, 768), (602112, 768, 1), 0); del buf475  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf476, buf477, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf478 = reinterpret_tensor(buf476, (6272, 768), (768, 1), 0); del buf476  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (6272, 768), (768, 1), 0), reinterpret_tensor(arg356_1, (768, 768), (1, 768), 0), out=buf478)
        del arg356_1
        buf482 = buf477; del buf477  # reuse
        # Source Nodes: [l__mod___blocks_12_norm3, mul_52, x_234, x_236], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf459, arg48_1, buf478, arg357_1, arg358_1, arg359_1, buf482, 6272, 768, grid=grid(6272), stream=stream0)
        del arg358_1
        del arg359_1
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(reinterpret_tensor(buf482, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg360_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf483, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg360_1
        buf484 = reinterpret_tensor(buf482, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf482  # reuse
        # Source Nodes: [x_238, x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf483, arg361_1, arg673_1, arg674_1, arg362_1, arg363_1, buf484, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg361_1
        del arg362_1
        del arg363_1
        del arg673_1
        del arg674_1
        # Source Nodes: [x_238, x_239, x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf485 = extern_kernels.convolution(buf484, arg364_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf485, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg364_1
        buf486 = reinterpret_tensor(buf484, (8, 784, 768), (602112, 768, 1), 0); del buf484  # reuse
        buf490 = reinterpret_tensor(buf483, (8, 784, 768), (602112, 768, 1), 0); del buf483  # reuse
        # Source Nodes: [l__mod___blocks_12_norm2, mul_52, mul_53, x_234, x_236, x_243], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf459, arg48_1, buf478, arg357_1, arg50_1, buf485, arg365_1, arg366_1, arg367_1, buf486, buf490, 6272, 768, grid=grid(6272), stream=stream0)
        del arg357_1
        del arg365_1
        del arg366_1
        del arg367_1
        del arg48_1
        del arg50_1
        del buf459
        buf491 = reinterpret_tensor(buf457, (6272, 3072), (3072, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (6272, 768), (768, 1), 0), reinterpret_tensor(arg368_1, (768, 3072), (1, 768), 0), out=buf491)
        del arg368_1
        buf492 = reinterpret_tensor(buf491, (8, 784, 3072), (2408448, 3072, 1), 0); del buf491  # reuse
        # Source Nodes: [x_245], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf492, arg369_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg369_1
        buf493 = reinterpret_tensor(buf490, (6272, 768), (768, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg370_1, (3072, 768), (1, 3072), 0), out=buf493)
        del arg370_1
        buf497 = reinterpret_tensor(buf485, (8, 784, 768), (602112, 768, 1), 0); del buf485  # reuse
        # Source Nodes: [l__mod___blocks_13_norm1, mul_54, x_251], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf486, arg51_1, buf493, arg371_1, arg372_1, arg373_1, buf497, 6272, 768, grid=grid(6272), stream=stream0)
        del arg372_1
        del arg373_1
        buf498 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf497, (6272, 768), (768, 1), 0), reinterpret_tensor(arg374_1, (768, 2304), (1, 768), 0), out=buf498)
        del arg374_1
        buf499 = buf467; del buf467  # reuse
        # Source Nodes: [q_27], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf498, arg375_1, buf499, 43008, 112, grid=grid(43008), stream=stream0)
        buf500 = buf468; del buf468  # reuse
        # Source Nodes: [q_27], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf499, buf500, 6144, 7, grid=grid(6144), stream=stream0)
        buf501 = buf499; del buf499  # reuse
        # Source Nodes: [k_27], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf498, arg375_1, buf501, 43008, 112, grid=grid(43008), stream=stream0)
        buf502 = buf466; del buf466  # reuse
        # Source Nodes: [k_27], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf501, buf502, 6144, 7, grid=grid(6144), stream=stream0)
        buf503 = reinterpret_tensor(buf497, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf497  # reuse
        # Source Nodes: [matmul_26, q_27], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf498, arg375_1, buf500, buf503, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf504 = reinterpret_tensor(buf478, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf478  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf498, arg375_1, buf502, buf504, 4816896, grid=grid(4816896), stream=stream0)
        buf505 = reinterpret_tensor(buf474, (128, 48, 48), (2304, 48, 1), 0); del buf474  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf503, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf504, (128, 784, 48), (37632, 48, 1), 0), out=buf505)
        buf508 = reinterpret_tensor(buf471, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf471  # reuse
        # Source Nodes: [attn_39, attn_40], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf505, arg53_1, buf508, 6144, 48, grid=grid(6144), stream=stream0)
        del arg53_1
        buf509 = reinterpret_tensor(buf504, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf504  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf498, arg375_1, buf509, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg375_1
        buf510 = reinterpret_tensor(buf503, (128, 48, 784), (37632, 784, 1), 0); del buf503  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf508, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf509, (128, 48, 784), (37632, 784, 1), 0), out=buf510)
        buf511 = reinterpret_tensor(buf509, (8, 784, 768), (602112, 768, 1), 0); del buf509  # reuse
        # Source Nodes: [x_253], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf510, buf511, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf512 = reinterpret_tensor(buf510, (6272, 768), (768, 1), 0); del buf510  # reuse
        # Source Nodes: [x_253], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (6272, 768), (768, 1), 0), reinterpret_tensor(arg376_1, (768, 768), (1, 768), 0), out=buf512)
        del arg376_1
        buf513 = reinterpret_tensor(buf512, (8, 784, 768), (602112, 768, 1), 0); del buf512  # reuse
        buf517 = buf511; del buf511  # reuse
        # Source Nodes: [l__mod___blocks_13_norm3, mul_54, mul_56, x_251, x_253, x_255], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf513, buf486, arg51_1, buf493, arg371_1, arg52_1, arg377_1, arg378_1, arg379_1, buf517, 6272, 768, grid=grid(6272), stream=stream0)
        del arg371_1
        del arg377_1
        del arg378_1
        del arg379_1
        del arg51_1
        del arg52_1
        del buf486
        del buf493
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(reinterpret_tensor(buf517, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg380_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf518, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg380_1
        buf519 = reinterpret_tensor(buf517, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf517  # reuse
        # Source Nodes: [x_257, x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf518, arg381_1, arg676_1, arg677_1, arg382_1, arg383_1, buf519, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg381_1
        del arg382_1
        del arg383_1
        del arg676_1
        del arg677_1
        # Source Nodes: [x_257, x_258, x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf520 = extern_kernels.convolution(buf519, arg384_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf520, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg384_1
        buf521 = buf451; del buf451  # reuse
        buf522 = buf450; del buf450  # reuse
        buf523 = buf449; del buf449  # reuse
        # Source Nodes: [l__mod___blocks_13_norm2, mul_57, x_262], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf513, arg54_1, buf520, arg385_1, buf521, buf522, buf523, 37632, 128, grid=grid(37632), stream=stream0)
        buf524 = buf453; del buf453  # reuse
        buf525 = buf452; del buf452  # reuse
        # Source Nodes: [l__mod___blocks_13_norm2, mul_57, x_262], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf521, buf522, buf523, buf524, buf525, 6272, 6, grid=grid(6272), stream=stream0)
        buf527 = reinterpret_tensor(buf519, (8, 784, 768), (602112, 768, 1), 0); del buf519  # reuse
        # Source Nodes: [l__mod___blocks_13_norm2, mul_57, x_262], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf513, arg54_1, buf520, arg385_1, buf524, buf525, arg386_1, arg387_1, buf527, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg386_1
        del arg387_1
        buf528 = reinterpret_tensor(buf492, (6272, 3072), (3072, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (6272, 768), (768, 1), 0), reinterpret_tensor(arg388_1, (768, 3072), (1, 768), 0), out=buf528)
        del arg388_1
        buf529 = reinterpret_tensor(buf528, (8, 784, 3072), (2408448, 3072, 1), 0); del buf528  # reuse
        # Source Nodes: [x_264], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf529, arg389_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg389_1
        buf530 = reinterpret_tensor(buf527, (6272, 768), (768, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg390_1, (3072, 768), (1, 3072), 0), out=buf530)
        del arg390_1
        buf531 = reinterpret_tensor(buf530, (8, 784, 768), (602112, 768, 1), 0); del buf530  # reuse
        buf535 = reinterpret_tensor(buf518, (8, 784, 768), (602112, 768, 1), 0); del buf518  # reuse
        # Source Nodes: [l__mod___blocks_14_norm1, mul_57, mul_58, x_262, x_270], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf531, buf513, arg54_1, buf520, arg385_1, arg55_1, arg391_1, arg392_1, arg393_1, buf535, 6272, 768, grid=grid(6272), stream=stream0)
        del arg385_1
        del arg391_1
        del arg392_1
        del arg393_1
        del arg54_1
        del arg55_1
        del buf513
        buf536 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf535, (6272, 768), (768, 1), 0), reinterpret_tensor(arg394_1, (768, 2304), (1, 768), 0), out=buf536)
        del arg394_1
        buf537 = buf501; del buf501  # reuse
        # Source Nodes: [q_29], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf536, arg395_1, buf537, 43008, 112, grid=grid(43008), stream=stream0)
        buf538 = buf502; del buf502  # reuse
        # Source Nodes: [q_29], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf537, buf538, 6144, 7, grid=grid(6144), stream=stream0)
        buf539 = buf537; del buf537  # reuse
        # Source Nodes: [k_29], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf536, arg395_1, buf539, 43008, 112, grid=grid(43008), stream=stream0)
        buf540 = buf500; del buf500  # reuse
        # Source Nodes: [k_29], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf539, buf540, 6144, 7, grid=grid(6144), stream=stream0)
        buf541 = reinterpret_tensor(buf535, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf535  # reuse
        # Source Nodes: [matmul_28, q_29], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf536, arg395_1, buf538, buf541, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf542 = reinterpret_tensor(buf520, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf520  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf536, arg395_1, buf540, buf542, 4816896, grid=grid(4816896), stream=stream0)
        buf543 = reinterpret_tensor(buf508, (128, 48, 48), (2304, 48, 1), 0); del buf508  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf541, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf542, (128, 784, 48), (37632, 48, 1), 0), out=buf543)
        buf546 = reinterpret_tensor(buf505, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf505  # reuse
        # Source Nodes: [attn_42, attn_43], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf543, arg57_1, buf546, 6144, 48, grid=grid(6144), stream=stream0)
        del arg57_1
        buf547 = reinterpret_tensor(buf542, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf542  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf536, arg395_1, buf547, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg395_1
        buf548 = reinterpret_tensor(buf541, (128, 48, 784), (37632, 784, 1), 0); del buf541  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf546, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf547, (128, 48, 784), (37632, 784, 1), 0), out=buf548)
        buf549 = reinterpret_tensor(buf547, (8, 784, 768), (602112, 768, 1), 0); del buf547  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf548, buf549, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf550 = reinterpret_tensor(buf548, (6272, 768), (768, 1), 0); del buf548  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (6272, 768), (768, 1), 0), reinterpret_tensor(arg396_1, (768, 768), (1, 768), 0), out=buf550)
        del arg396_1
        buf554 = buf549; del buf549  # reuse
        # Source Nodes: [l__mod___blocks_14_norm3, mul_60, x_272, x_274], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf531, arg56_1, buf550, arg397_1, arg398_1, arg399_1, buf554, 6272, 768, grid=grid(6272), stream=stream0)
        del arg398_1
        del arg399_1
        # Source Nodes: [x_276], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(reinterpret_tensor(buf554, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg400_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf555, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg400_1
        buf556 = reinterpret_tensor(buf554, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf554  # reuse
        # Source Nodes: [x_276, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf555, arg401_1, arg679_1, arg680_1, arg402_1, arg403_1, buf556, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg401_1
        del arg402_1
        del arg403_1
        del arg679_1
        del arg680_1
        # Source Nodes: [x_276, x_277, x_278, x_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf557 = extern_kernels.convolution(buf556, arg404_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf557, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg404_1
        buf558 = reinterpret_tensor(buf556, (8, 784, 768), (602112, 768, 1), 0); del buf556  # reuse
        buf562 = reinterpret_tensor(buf555, (8, 784, 768), (602112, 768, 1), 0); del buf555  # reuse
        # Source Nodes: [l__mod___blocks_14_norm2, mul_60, mul_61, x_272, x_274, x_281], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf531, arg56_1, buf550, arg397_1, arg58_1, buf557, arg405_1, arg406_1, arg407_1, buf558, buf562, 6272, 768, grid=grid(6272), stream=stream0)
        del arg397_1
        del arg405_1
        del arg406_1
        del arg407_1
        del arg56_1
        del arg58_1
        del buf531
        buf563 = reinterpret_tensor(buf529, (6272, 3072), (3072, 1), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (6272, 768), (768, 1), 0), reinterpret_tensor(arg408_1, (768, 3072), (1, 768), 0), out=buf563)
        del arg408_1
        buf564 = reinterpret_tensor(buf563, (8, 784, 3072), (2408448, 3072, 1), 0); del buf563  # reuse
        # Source Nodes: [x_283], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf564, arg409_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg409_1
        buf565 = reinterpret_tensor(buf562, (6272, 768), (768, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg410_1, (3072, 768), (1, 3072), 0), out=buf565)
        del arg410_1
        buf569 = reinterpret_tensor(buf557, (8, 784, 768), (602112, 768, 1), 0); del buf557  # reuse
        # Source Nodes: [l__mod___blocks_15_norm1, mul_62, x_289], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf558, arg59_1, buf565, arg411_1, arg412_1, arg413_1, buf569, 6272, 768, grid=grid(6272), stream=stream0)
        del arg412_1
        del arg413_1
        buf570 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf569, (6272, 768), (768, 1), 0), reinterpret_tensor(arg414_1, (768, 2304), (1, 768), 0), out=buf570)
        del arg414_1
        buf571 = buf539; del buf539  # reuse
        # Source Nodes: [q_31], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf570, arg415_1, buf571, 43008, 112, grid=grid(43008), stream=stream0)
        buf572 = buf540; del buf540  # reuse
        # Source Nodes: [q_31], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf571, buf572, 6144, 7, grid=grid(6144), stream=stream0)
        buf573 = buf571; del buf571  # reuse
        # Source Nodes: [k_31], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf570, arg415_1, buf573, 43008, 112, grid=grid(43008), stream=stream0)
        buf574 = buf538; del buf538  # reuse
        # Source Nodes: [k_31], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf573, buf574, 6144, 7, grid=grid(6144), stream=stream0)
        buf575 = reinterpret_tensor(buf569, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf569  # reuse
        # Source Nodes: [matmul_30, q_31], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf570, arg415_1, buf572, buf575, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf576 = reinterpret_tensor(buf550, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf550  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf570, arg415_1, buf574, buf576, 4816896, grid=grid(4816896), stream=stream0)
        buf577 = reinterpret_tensor(buf546, (128, 48, 48), (2304, 48, 1), 0); del buf546  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf575, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf576, (128, 784, 48), (37632, 48, 1), 0), out=buf577)
        buf580 = reinterpret_tensor(buf543, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf543  # reuse
        # Source Nodes: [attn_45, attn_46], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf577, arg61_1, buf580, 6144, 48, grid=grid(6144), stream=stream0)
        del arg61_1
        buf581 = reinterpret_tensor(buf576, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf576  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf570, arg415_1, buf581, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg415_1
        buf582 = reinterpret_tensor(buf575, (128, 48, 784), (37632, 784, 1), 0); del buf575  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf580, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf581, (128, 48, 784), (37632, 784, 1), 0), out=buf582)
        buf583 = reinterpret_tensor(buf581, (8, 784, 768), (602112, 768, 1), 0); del buf581  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf582, buf583, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf584 = reinterpret_tensor(buf582, (6272, 768), (768, 1), 0); del buf582  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (6272, 768), (768, 1), 0), reinterpret_tensor(arg416_1, (768, 768), (1, 768), 0), out=buf584)
        del arg416_1
        buf585 = reinterpret_tensor(buf584, (8, 784, 768), (602112, 768, 1), 0); del buf584  # reuse
        buf589 = buf583; del buf583  # reuse
        # Source Nodes: [l__mod___blocks_15_norm3, mul_62, mul_64, x_289, x_291, x_293], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf585, buf558, arg59_1, buf565, arg411_1, arg60_1, arg417_1, arg418_1, arg419_1, buf589, 6272, 768, grid=grid(6272), stream=stream0)
        del arg411_1
        del arg417_1
        del arg418_1
        del arg419_1
        del arg59_1
        del arg60_1
        del buf558
        del buf565
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(reinterpret_tensor(buf589, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg420_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf590, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg420_1
        buf591 = reinterpret_tensor(buf589, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf589  # reuse
        # Source Nodes: [x_295, x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf590, arg421_1, arg682_1, arg683_1, arg422_1, arg423_1, buf591, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg421_1
        del arg422_1
        del arg423_1
        del arg682_1
        del arg683_1
        # Source Nodes: [x_295, x_296, x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf592 = extern_kernels.convolution(buf591, arg424_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf592, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg424_1
        buf593 = buf523; del buf523  # reuse
        buf594 = buf522; del buf522  # reuse
        buf595 = buf521; del buf521  # reuse
        # Source Nodes: [l__mod___blocks_15_norm2, mul_65, x_300], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf585, arg62_1, buf592, arg425_1, buf593, buf594, buf595, 37632, 128, grid=grid(37632), stream=stream0)
        buf596 = buf525; del buf525  # reuse
        buf597 = buf524; del buf524  # reuse
        # Source Nodes: [l__mod___blocks_15_norm2, mul_65, x_300], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf593, buf594, buf595, buf596, buf597, 6272, 6, grid=grid(6272), stream=stream0)
        buf599 = reinterpret_tensor(buf591, (8, 784, 768), (602112, 768, 1), 0); del buf591  # reuse
        # Source Nodes: [l__mod___blocks_15_norm2, mul_65, x_300], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf585, arg62_1, buf592, arg425_1, buf596, buf597, arg426_1, arg427_1, buf599, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg426_1
        del arg427_1
        buf600 = reinterpret_tensor(buf564, (6272, 3072), (3072, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (6272, 768), (768, 1), 0), reinterpret_tensor(arg428_1, (768, 3072), (1, 768), 0), out=buf600)
        del arg428_1
        buf601 = reinterpret_tensor(buf600, (8, 784, 3072), (2408448, 3072, 1), 0); del buf600  # reuse
        # Source Nodes: [x_302], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf601, arg429_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg429_1
        buf602 = reinterpret_tensor(buf599, (6272, 768), (768, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf601, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg430_1, (3072, 768), (1, 3072), 0), out=buf602)
        del arg430_1
        buf603 = reinterpret_tensor(buf602, (8, 784, 768), (602112, 768, 1), 0); del buf602  # reuse
        buf607 = reinterpret_tensor(buf590, (8, 784, 768), (602112, 768, 1), 0); del buf590  # reuse
        # Source Nodes: [l__mod___blocks_16_norm1, mul_65, mul_66, x_300, x_308], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf603, buf585, arg62_1, buf592, arg425_1, arg63_1, arg431_1, arg432_1, arg433_1, buf607, 6272, 768, grid=grid(6272), stream=stream0)
        del arg425_1
        del arg431_1
        del arg432_1
        del arg433_1
        del arg62_1
        del arg63_1
        del buf585
        buf608 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf607, (6272, 768), (768, 1), 0), reinterpret_tensor(arg434_1, (768, 2304), (1, 768), 0), out=buf608)
        del arg434_1
        buf609 = buf573; del buf573  # reuse
        # Source Nodes: [q_33], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf608, arg435_1, buf609, 43008, 112, grid=grid(43008), stream=stream0)
        buf610 = buf574; del buf574  # reuse
        # Source Nodes: [q_33], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf609, buf610, 6144, 7, grid=grid(6144), stream=stream0)
        buf611 = buf609; del buf609  # reuse
        # Source Nodes: [k_33], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf608, arg435_1, buf611, 43008, 112, grid=grid(43008), stream=stream0)
        buf612 = buf572; del buf572  # reuse
        # Source Nodes: [k_33], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf611, buf612, 6144, 7, grid=grid(6144), stream=stream0)
        buf613 = reinterpret_tensor(buf607, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf607  # reuse
        # Source Nodes: [matmul_32, q_33], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf608, arg435_1, buf610, buf613, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf614 = reinterpret_tensor(buf592, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf592  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf608, arg435_1, buf612, buf614, 4816896, grid=grid(4816896), stream=stream0)
        buf615 = reinterpret_tensor(buf580, (128, 48, 48), (2304, 48, 1), 0); del buf580  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf613, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf614, (128, 784, 48), (37632, 48, 1), 0), out=buf615)
        buf618 = reinterpret_tensor(buf577, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf577  # reuse
        # Source Nodes: [attn_48, attn_49], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf615, arg65_1, buf618, 6144, 48, grid=grid(6144), stream=stream0)
        del arg65_1
        buf619 = reinterpret_tensor(buf614, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf614  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf608, arg435_1, buf619, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg435_1
        buf620 = reinterpret_tensor(buf613, (128, 48, 784), (37632, 784, 1), 0); del buf613  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf618, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf619, (128, 48, 784), (37632, 784, 1), 0), out=buf620)
        buf621 = reinterpret_tensor(buf619, (8, 784, 768), (602112, 768, 1), 0); del buf619  # reuse
        # Source Nodes: [x_310], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf620, buf621, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf622 = reinterpret_tensor(buf620, (6272, 768), (768, 1), 0); del buf620  # reuse
        # Source Nodes: [x_310], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (6272, 768), (768, 1), 0), reinterpret_tensor(arg436_1, (768, 768), (1, 768), 0), out=buf622)
        del arg436_1
        buf626 = buf621; del buf621  # reuse
        # Source Nodes: [l__mod___blocks_16_norm3, mul_68, x_310, x_312], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf603, arg64_1, buf622, arg437_1, arg438_1, arg439_1, buf626, 6272, 768, grid=grid(6272), stream=stream0)
        del arg438_1
        del arg439_1
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(reinterpret_tensor(buf626, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg440_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf627, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg440_1
        buf628 = reinterpret_tensor(buf626, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf626  # reuse
        # Source Nodes: [x_314, x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf627, arg441_1, arg685_1, arg686_1, arg442_1, arg443_1, buf628, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg441_1
        del arg442_1
        del arg443_1
        del arg685_1
        del arg686_1
        # Source Nodes: [x_314, x_315, x_316, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf629 = extern_kernels.convolution(buf628, arg444_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf629, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg444_1
        buf630 = reinterpret_tensor(buf628, (8, 784, 768), (602112, 768, 1), 0); del buf628  # reuse
        buf634 = reinterpret_tensor(buf627, (8, 784, 768), (602112, 768, 1), 0); del buf627  # reuse
        # Source Nodes: [l__mod___blocks_16_norm2, mul_68, mul_69, x_310, x_312, x_319], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf603, arg64_1, buf622, arg437_1, arg66_1, buf629, arg445_1, arg446_1, arg447_1, buf630, buf634, 6272, 768, grid=grid(6272), stream=stream0)
        del arg437_1
        del arg445_1
        del arg446_1
        del arg447_1
        del arg64_1
        del arg66_1
        del buf603
        buf635 = reinterpret_tensor(buf601, (6272, 3072), (3072, 1), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf634, (6272, 768), (768, 1), 0), reinterpret_tensor(arg448_1, (768, 3072), (1, 768), 0), out=buf635)
        del arg448_1
        buf636 = reinterpret_tensor(buf635, (8, 784, 3072), (2408448, 3072, 1), 0); del buf635  # reuse
        # Source Nodes: [x_321], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf636, arg449_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg449_1
        buf637 = reinterpret_tensor(buf634, (6272, 768), (768, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf636, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg450_1, (3072, 768), (1, 3072), 0), out=buf637)
        del arg450_1
        buf641 = reinterpret_tensor(buf629, (8, 784, 768), (602112, 768, 1), 0); del buf629  # reuse
        # Source Nodes: [l__mod___blocks_17_norm1, mul_70, x_327], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf630, arg67_1, buf637, arg451_1, arg452_1, arg453_1, buf641, 6272, 768, grid=grid(6272), stream=stream0)
        del arg452_1
        del arg453_1
        buf642 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf641, (6272, 768), (768, 1), 0), reinterpret_tensor(arg454_1, (768, 2304), (1, 768), 0), out=buf642)
        del arg454_1
        buf643 = buf611; del buf611  # reuse
        # Source Nodes: [q_35], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf642, arg455_1, buf643, 43008, 112, grid=grid(43008), stream=stream0)
        buf644 = buf612; del buf612  # reuse
        # Source Nodes: [q_35], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf643, buf644, 6144, 7, grid=grid(6144), stream=stream0)
        buf645 = buf643; del buf643  # reuse
        # Source Nodes: [k_35], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf642, arg455_1, buf645, 43008, 112, grid=grid(43008), stream=stream0)
        buf646 = buf610; del buf610  # reuse
        # Source Nodes: [k_35], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf645, buf646, 6144, 7, grid=grid(6144), stream=stream0)
        buf647 = reinterpret_tensor(buf641, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf641  # reuse
        # Source Nodes: [matmul_34, q_35], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf642, arg455_1, buf644, buf647, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf648 = reinterpret_tensor(buf622, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf622  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf642, arg455_1, buf646, buf648, 4816896, grid=grid(4816896), stream=stream0)
        buf649 = reinterpret_tensor(buf618, (128, 48, 48), (2304, 48, 1), 0); del buf618  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf647, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf648, (128, 784, 48), (37632, 48, 1), 0), out=buf649)
        buf652 = reinterpret_tensor(buf615, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf615  # reuse
        # Source Nodes: [attn_51, attn_52], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf649, arg69_1, buf652, 6144, 48, grid=grid(6144), stream=stream0)
        del arg69_1
        buf653 = reinterpret_tensor(buf648, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf648  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf642, arg455_1, buf653, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg455_1
        buf654 = reinterpret_tensor(buf647, (128, 48, 784), (37632, 784, 1), 0); del buf647  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf652, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf653, (128, 48, 784), (37632, 784, 1), 0), out=buf654)
        buf655 = reinterpret_tensor(buf653, (8, 784, 768), (602112, 768, 1), 0); del buf653  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf654, buf655, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf656 = reinterpret_tensor(buf654, (6272, 768), (768, 1), 0); del buf654  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (6272, 768), (768, 1), 0), reinterpret_tensor(arg456_1, (768, 768), (1, 768), 0), out=buf656)
        del arg456_1
        buf657 = reinterpret_tensor(buf656, (8, 784, 768), (602112, 768, 1), 0); del buf656  # reuse
        buf661 = buf655; del buf655  # reuse
        # Source Nodes: [l__mod___blocks_17_norm3, mul_70, mul_72, x_327, x_329, x_331], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf657, buf630, arg67_1, buf637, arg451_1, arg68_1, arg457_1, arg458_1, arg459_1, buf661, 6272, 768, grid=grid(6272), stream=stream0)
        del arg451_1
        del arg457_1
        del arg458_1
        del arg459_1
        del arg67_1
        del arg68_1
        del buf630
        del buf637
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(reinterpret_tensor(buf661, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg460_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf662, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg460_1
        buf663 = reinterpret_tensor(buf661, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf661  # reuse
        # Source Nodes: [x_333, x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf662, arg461_1, arg688_1, arg689_1, arg462_1, arg463_1, buf663, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg461_1
        del arg462_1
        del arg463_1
        del arg688_1
        del arg689_1
        # Source Nodes: [x_333, x_334, x_335, x_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf664 = extern_kernels.convolution(buf663, arg464_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf664, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg464_1
        buf665 = buf595; del buf595  # reuse
        buf666 = buf594; del buf594  # reuse
        buf667 = buf593; del buf593  # reuse
        # Source Nodes: [l__mod___blocks_17_norm2, mul_73, x_338], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf657, arg70_1, buf664, arg465_1, buf665, buf666, buf667, 37632, 128, grid=grid(37632), stream=stream0)
        buf668 = buf597; del buf597  # reuse
        buf669 = buf596; del buf596  # reuse
        # Source Nodes: [l__mod___blocks_17_norm2, mul_73, x_338], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf665, buf666, buf667, buf668, buf669, 6272, 6, grid=grid(6272), stream=stream0)
        buf671 = reinterpret_tensor(buf663, (8, 784, 768), (602112, 768, 1), 0); del buf663  # reuse
        # Source Nodes: [l__mod___blocks_17_norm2, mul_73, x_338], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf657, arg70_1, buf664, arg465_1, buf668, buf669, arg466_1, arg467_1, buf671, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg466_1
        del arg467_1
        buf672 = reinterpret_tensor(buf636, (6272, 3072), (3072, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf671, (6272, 768), (768, 1), 0), reinterpret_tensor(arg468_1, (768, 3072), (1, 768), 0), out=buf672)
        del arg468_1
        buf673 = reinterpret_tensor(buf672, (8, 784, 3072), (2408448, 3072, 1), 0); del buf672  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf673, arg469_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg469_1
        buf674 = reinterpret_tensor(buf671, (6272, 768), (768, 1), 0); del buf671  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf673, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg470_1, (3072, 768), (1, 3072), 0), out=buf674)
        del arg470_1
        buf675 = reinterpret_tensor(buf674, (8, 784, 768), (602112, 768, 1), 0); del buf674  # reuse
        buf679 = reinterpret_tensor(buf662, (8, 784, 768), (602112, 768, 1), 0); del buf662  # reuse
        # Source Nodes: [l__mod___blocks_18_norm1, mul_73, mul_74, x_338, x_346], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf675, buf657, arg70_1, buf664, arg465_1, arg71_1, arg471_1, arg472_1, arg473_1, buf679, 6272, 768, grid=grid(6272), stream=stream0)
        del arg465_1
        del arg471_1
        del arg472_1
        del arg473_1
        del arg70_1
        del arg71_1
        del buf657
        buf680 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf679, (6272, 768), (768, 1), 0), reinterpret_tensor(arg474_1, (768, 2304), (1, 768), 0), out=buf680)
        del arg474_1
        buf681 = buf645; del buf645  # reuse
        # Source Nodes: [q_37], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf680, arg475_1, buf681, 43008, 112, grid=grid(43008), stream=stream0)
        buf682 = buf646; del buf646  # reuse
        # Source Nodes: [q_37], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf681, buf682, 6144, 7, grid=grid(6144), stream=stream0)
        buf683 = buf681; del buf681  # reuse
        # Source Nodes: [k_37], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf680, arg475_1, buf683, 43008, 112, grid=grid(43008), stream=stream0)
        buf684 = buf644; del buf644  # reuse
        # Source Nodes: [k_37], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf683, buf684, 6144, 7, grid=grid(6144), stream=stream0)
        buf685 = reinterpret_tensor(buf679, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf679  # reuse
        # Source Nodes: [matmul_36, q_37], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf680, arg475_1, buf682, buf685, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf686 = reinterpret_tensor(buf664, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf664  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf680, arg475_1, buf684, buf686, 4816896, grid=grid(4816896), stream=stream0)
        buf687 = reinterpret_tensor(buf652, (128, 48, 48), (2304, 48, 1), 0); del buf652  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf685, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf686, (128, 784, 48), (37632, 48, 1), 0), out=buf687)
        buf690 = reinterpret_tensor(buf649, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf649  # reuse
        # Source Nodes: [attn_54, attn_55], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf687, arg73_1, buf690, 6144, 48, grid=grid(6144), stream=stream0)
        del arg73_1
        buf691 = reinterpret_tensor(buf686, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf686  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf680, arg475_1, buf691, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg475_1
        buf692 = reinterpret_tensor(buf685, (128, 48, 784), (37632, 784, 1), 0); del buf685  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf690, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf691, (128, 48, 784), (37632, 784, 1), 0), out=buf692)
        buf693 = reinterpret_tensor(buf691, (8, 784, 768), (602112, 768, 1), 0); del buf691  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf692, buf693, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf694 = reinterpret_tensor(buf692, (6272, 768), (768, 1), 0); del buf692  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (6272, 768), (768, 1), 0), reinterpret_tensor(arg476_1, (768, 768), (1, 768), 0), out=buf694)
        del arg476_1
        buf698 = buf693; del buf693  # reuse
        # Source Nodes: [l__mod___blocks_18_norm3, mul_76, x_348, x_350], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf675, arg72_1, buf694, arg477_1, arg478_1, arg479_1, buf698, 6272, 768, grid=grid(6272), stream=stream0)
        del arg478_1
        del arg479_1
        # Source Nodes: [x_352], Original ATen: [aten.convolution]
        buf699 = extern_kernels.convolution(reinterpret_tensor(buf698, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg480_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf699, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg480_1
        buf700 = reinterpret_tensor(buf698, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf698  # reuse
        # Source Nodes: [x_352, x_353, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf699, arg481_1, arg691_1, arg692_1, arg482_1, arg483_1, buf700, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg481_1
        del arg482_1
        del arg483_1
        del arg691_1
        del arg692_1
        # Source Nodes: [x_352, x_353, x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf701 = extern_kernels.convolution(buf700, arg484_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf701, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg484_1
        buf702 = reinterpret_tensor(buf700, (8, 784, 768), (602112, 768, 1), 0); del buf700  # reuse
        buf706 = reinterpret_tensor(buf699, (8, 784, 768), (602112, 768, 1), 0); del buf699  # reuse
        # Source Nodes: [l__mod___blocks_18_norm2, mul_76, mul_77, x_348, x_350, x_357], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf675, arg72_1, buf694, arg477_1, arg74_1, buf701, arg485_1, arg486_1, arg487_1, buf702, buf706, 6272, 768, grid=grid(6272), stream=stream0)
        del arg477_1
        del arg485_1
        del arg486_1
        del arg487_1
        del arg72_1
        del arg74_1
        del buf675
        buf707 = reinterpret_tensor(buf673, (6272, 3072), (3072, 1), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf706, (6272, 768), (768, 1), 0), reinterpret_tensor(arg488_1, (768, 3072), (1, 768), 0), out=buf707)
        del arg488_1
        buf708 = reinterpret_tensor(buf707, (8, 784, 3072), (2408448, 3072, 1), 0); del buf707  # reuse
        # Source Nodes: [x_359], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf708, arg489_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg489_1
        buf709 = reinterpret_tensor(buf706, (6272, 768), (768, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf708, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg490_1, (3072, 768), (1, 3072), 0), out=buf709)
        del arg490_1
        buf713 = reinterpret_tensor(buf701, (8, 784, 768), (602112, 768, 1), 0); del buf701  # reuse
        # Source Nodes: [l__mod___blocks_19_norm1, mul_78, x_365], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf702, arg75_1, buf709, arg491_1, arg492_1, arg493_1, buf713, 6272, 768, grid=grid(6272), stream=stream0)
        del arg492_1
        del arg493_1
        buf714 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf713, (6272, 768), (768, 1), 0), reinterpret_tensor(arg494_1, (768, 2304), (1, 768), 0), out=buf714)
        del arg494_1
        buf715 = buf683; del buf683  # reuse
        # Source Nodes: [q_39], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf714, arg495_1, buf715, 43008, 112, grid=grid(43008), stream=stream0)
        buf716 = buf684; del buf684  # reuse
        # Source Nodes: [q_39], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf715, buf716, 6144, 7, grid=grid(6144), stream=stream0)
        buf717 = buf715; del buf715  # reuse
        # Source Nodes: [k_39], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf714, arg495_1, buf717, 43008, 112, grid=grid(43008), stream=stream0)
        buf718 = buf682; del buf682  # reuse
        # Source Nodes: [k_39], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf717, buf718, 6144, 7, grid=grid(6144), stream=stream0)
        buf719 = reinterpret_tensor(buf713, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf713  # reuse
        # Source Nodes: [matmul_38, q_39], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf714, arg495_1, buf716, buf719, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf720 = reinterpret_tensor(buf694, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf694  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf714, arg495_1, buf718, buf720, 4816896, grid=grid(4816896), stream=stream0)
        buf721 = reinterpret_tensor(buf690, (128, 48, 48), (2304, 48, 1), 0); del buf690  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf719, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf720, (128, 784, 48), (37632, 48, 1), 0), out=buf721)
        buf724 = reinterpret_tensor(buf687, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf687  # reuse
        # Source Nodes: [attn_57, attn_58], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf721, arg77_1, buf724, 6144, 48, grid=grid(6144), stream=stream0)
        del arg77_1
        buf725 = reinterpret_tensor(buf720, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf720  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf714, arg495_1, buf725, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg495_1
        buf726 = reinterpret_tensor(buf719, (128, 48, 784), (37632, 784, 1), 0); del buf719  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf724, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf725, (128, 48, 784), (37632, 784, 1), 0), out=buf726)
        buf727 = reinterpret_tensor(buf725, (8, 784, 768), (602112, 768, 1), 0); del buf725  # reuse
        # Source Nodes: [x_367], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf726, buf727, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf728 = reinterpret_tensor(buf726, (6272, 768), (768, 1), 0); del buf726  # reuse
        # Source Nodes: [x_367], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (6272, 768), (768, 1), 0), reinterpret_tensor(arg496_1, (768, 768), (1, 768), 0), out=buf728)
        del arg496_1
        buf729 = reinterpret_tensor(buf728, (8, 784, 768), (602112, 768, 1), 0); del buf728  # reuse
        buf733 = buf727; del buf727  # reuse
        # Source Nodes: [l__mod___blocks_19_norm3, mul_78, mul_80, x_365, x_367, x_369], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf729, buf702, arg75_1, buf709, arg491_1, arg76_1, arg497_1, arg498_1, arg499_1, buf733, 6272, 768, grid=grid(6272), stream=stream0)
        del arg491_1
        del arg497_1
        del arg498_1
        del arg499_1
        del arg75_1
        del arg76_1
        del buf702
        del buf709
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf734 = extern_kernels.convolution(reinterpret_tensor(buf733, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg500_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf734, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg500_1
        buf735 = reinterpret_tensor(buf733, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf733  # reuse
        # Source Nodes: [x_371, x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf734, arg501_1, arg694_1, arg695_1, arg502_1, arg503_1, buf735, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg501_1
        del arg502_1
        del arg503_1
        del arg694_1
        del arg695_1
        # Source Nodes: [x_371, x_372, x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf736 = extern_kernels.convolution(buf735, arg504_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf736, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg504_1
        buf737 = buf667; del buf667  # reuse
        buf738 = buf666; del buf666  # reuse
        buf739 = buf665; del buf665  # reuse
        # Source Nodes: [l__mod___blocks_19_norm2, mul_81, x_376], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf729, arg78_1, buf736, arg505_1, buf737, buf738, buf739, 37632, 128, grid=grid(37632), stream=stream0)
        buf740 = buf669; del buf669  # reuse
        buf741 = buf668; del buf668  # reuse
        # Source Nodes: [l__mod___blocks_19_norm2, mul_81, x_376], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf737, buf738, buf739, buf740, buf741, 6272, 6, grid=grid(6272), stream=stream0)
        buf743 = reinterpret_tensor(buf735, (8, 784, 768), (602112, 768, 1), 0); del buf735  # reuse
        # Source Nodes: [l__mod___blocks_19_norm2, mul_81, x_376], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf729, arg78_1, buf736, arg505_1, buf740, buf741, arg506_1, arg507_1, buf743, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg506_1
        del arg507_1
        buf744 = reinterpret_tensor(buf708, (6272, 3072), (3072, 1), 0); del buf708  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf743, (6272, 768), (768, 1), 0), reinterpret_tensor(arg508_1, (768, 3072), (1, 768), 0), out=buf744)
        del arg508_1
        buf745 = reinterpret_tensor(buf744, (8, 784, 3072), (2408448, 3072, 1), 0); del buf744  # reuse
        # Source Nodes: [x_378], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf745, arg509_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg509_1
        buf746 = reinterpret_tensor(buf743, (6272, 768), (768, 1), 0); del buf743  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf745, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg510_1, (3072, 768), (1, 3072), 0), out=buf746)
        del arg510_1
        buf747 = reinterpret_tensor(buf746, (8, 784, 768), (602112, 768, 1), 0); del buf746  # reuse
        buf751 = reinterpret_tensor(buf734, (8, 784, 768), (602112, 768, 1), 0); del buf734  # reuse
        # Source Nodes: [l__mod___blocks_20_norm1, mul_81, mul_82, x_376, x_384], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf747, buf729, arg78_1, buf736, arg505_1, arg79_1, arg511_1, arg512_1, arg513_1, buf751, 6272, 768, grid=grid(6272), stream=stream0)
        del arg505_1
        del arg511_1
        del arg512_1
        del arg513_1
        del arg78_1
        del arg79_1
        del buf729
        buf752 = buf714; del buf714  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf751, (6272, 768), (768, 1), 0), reinterpret_tensor(arg514_1, (768, 2304), (1, 768), 0), out=buf752)
        del arg514_1
        buf753 = buf717; del buf717  # reuse
        # Source Nodes: [q_41], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf752, arg515_1, buf753, 43008, 112, grid=grid(43008), stream=stream0)
        buf754 = buf718; del buf718  # reuse
        # Source Nodes: [q_41], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf753, buf754, 6144, 7, grid=grid(6144), stream=stream0)
        buf755 = buf753; del buf753  # reuse
        # Source Nodes: [k_41], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf752, arg515_1, buf755, 43008, 112, grid=grid(43008), stream=stream0)
        buf756 = buf716; del buf716  # reuse
        # Source Nodes: [k_41], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf755, buf756, 6144, 7, grid=grid(6144), stream=stream0)
        buf757 = reinterpret_tensor(buf751, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf751  # reuse
        # Source Nodes: [matmul_40, q_41], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf752, arg515_1, buf754, buf757, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf758 = reinterpret_tensor(buf736, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf736  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf752, arg515_1, buf756, buf758, 4816896, grid=grid(4816896), stream=stream0)
        buf759 = reinterpret_tensor(buf724, (128, 48, 48), (2304, 48, 1), 0); del buf724  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf757, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf758, (128, 784, 48), (37632, 48, 1), 0), out=buf759)
        buf762 = reinterpret_tensor(buf721, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf721  # reuse
        # Source Nodes: [attn_60, attn_61], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf759, arg81_1, buf762, 6144, 48, grid=grid(6144), stream=stream0)
        del arg81_1
        buf763 = reinterpret_tensor(buf758, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf758  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf752, arg515_1, buf763, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg515_1
        buf764 = reinterpret_tensor(buf757, (128, 48, 784), (37632, 784, 1), 0); del buf757  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf762, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf763, (128, 48, 784), (37632, 784, 1), 0), out=buf764)
        buf765 = reinterpret_tensor(buf763, (8, 784, 768), (602112, 768, 1), 0); del buf763  # reuse
        # Source Nodes: [x_386], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf764, buf765, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf766 = reinterpret_tensor(buf764, (6272, 768), (768, 1), 0); del buf764  # reuse
        # Source Nodes: [x_386], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (6272, 768), (768, 1), 0), reinterpret_tensor(arg516_1, (768, 768), (1, 768), 0), out=buf766)
        del arg516_1
        buf770 = buf765; del buf765  # reuse
        # Source Nodes: [l__mod___blocks_20_norm3, mul_84, x_386, x_388], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf747, arg80_1, buf766, arg517_1, arg518_1, arg519_1, buf770, 6272, 768, grid=grid(6272), stream=stream0)
        del arg518_1
        del arg519_1
        # Source Nodes: [x_390], Original ATen: [aten.convolution]
        buf771 = extern_kernels.convolution(reinterpret_tensor(buf770, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg520_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf771, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg520_1
        buf772 = reinterpret_tensor(buf770, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf770  # reuse
        # Source Nodes: [x_390, x_391, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf771, arg521_1, arg697_1, arg698_1, arg522_1, arg523_1, buf772, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg521_1
        del arg522_1
        del arg523_1
        del arg697_1
        del arg698_1
        # Source Nodes: [x_390, x_391, x_392, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf773 = extern_kernels.convolution(buf772, arg524_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf773, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg524_1
        buf774 = reinterpret_tensor(buf772, (8, 784, 768), (602112, 768, 1), 0); del buf772  # reuse
        buf778 = reinterpret_tensor(buf771, (8, 784, 768), (602112, 768, 1), 0); del buf771  # reuse
        # Source Nodes: [l__mod___blocks_20_norm2, mul_84, mul_85, x_386, x_388, x_395], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf747, arg80_1, buf766, arg517_1, arg82_1, buf773, arg525_1, arg526_1, arg527_1, buf774, buf778, 6272, 768, grid=grid(6272), stream=stream0)
        del arg517_1
        del arg525_1
        del arg526_1
        del arg527_1
        del arg80_1
        del arg82_1
        del buf747
        buf779 = reinterpret_tensor(buf745, (6272, 3072), (3072, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf778, (6272, 768), (768, 1), 0), reinterpret_tensor(arg528_1, (768, 3072), (1, 768), 0), out=buf779)
        del arg528_1
        buf780 = reinterpret_tensor(buf779, (8, 784, 3072), (2408448, 3072, 1), 0); del buf779  # reuse
        # Source Nodes: [x_397], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf780, arg529_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg529_1
        buf781 = reinterpret_tensor(buf778, (6272, 768), (768, 1), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf780, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg530_1, (3072, 768), (1, 3072), 0), out=buf781)
        del arg530_1
        buf785 = reinterpret_tensor(buf773, (8, 784, 768), (602112, 768, 1), 0); del buf773  # reuse
        # Source Nodes: [l__mod___blocks_21_norm1, mul_86, x_403], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf774, arg83_1, buf781, arg531_1, arg532_1, arg533_1, buf785, 6272, 768, grid=grid(6272), stream=stream0)
        del arg532_1
        del arg533_1
        buf786 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf785, (6272, 768), (768, 1), 0), reinterpret_tensor(arg534_1, (768, 2304), (1, 768), 0), out=buf786)
        del arg534_1
        buf787 = buf755; del buf755  # reuse
        # Source Nodes: [q_43], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf786, arg535_1, buf787, 43008, 112, grid=grid(43008), stream=stream0)
        buf788 = buf756; del buf756  # reuse
        # Source Nodes: [q_43], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf787, buf788, 6144, 7, grid=grid(6144), stream=stream0)
        buf789 = buf787; del buf787  # reuse
        # Source Nodes: [k_43], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf786, arg535_1, buf789, 43008, 112, grid=grid(43008), stream=stream0)
        buf790 = buf754; del buf754  # reuse
        # Source Nodes: [k_43], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf789, buf790, 6144, 7, grid=grid(6144), stream=stream0)
        buf791 = reinterpret_tensor(buf785, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf785  # reuse
        # Source Nodes: [matmul_42, q_43], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf786, arg535_1, buf788, buf791, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf792 = reinterpret_tensor(buf766, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf766  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf786, arg535_1, buf790, buf792, 4816896, grid=grid(4816896), stream=stream0)
        buf793 = reinterpret_tensor(buf762, (128, 48, 48), (2304, 48, 1), 0); del buf762  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf791, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf792, (128, 784, 48), (37632, 48, 1), 0), out=buf793)
        buf796 = reinterpret_tensor(buf759, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf759  # reuse
        # Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf793, arg85_1, buf796, 6144, 48, grid=grid(6144), stream=stream0)
        del arg85_1
        buf797 = reinterpret_tensor(buf792, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf792  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf786, arg535_1, buf797, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg535_1
        buf798 = reinterpret_tensor(buf791, (128, 48, 784), (37632, 784, 1), 0); del buf791  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf796, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf797, (128, 48, 784), (37632, 784, 1), 0), out=buf798)
        buf799 = reinterpret_tensor(buf797, (8, 784, 768), (602112, 768, 1), 0); del buf797  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf798, buf799, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf800 = reinterpret_tensor(buf798, (6272, 768), (768, 1), 0); del buf798  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf799, (6272, 768), (768, 1), 0), reinterpret_tensor(arg536_1, (768, 768), (1, 768), 0), out=buf800)
        del arg536_1
        buf801 = reinterpret_tensor(buf800, (8, 784, 768), (602112, 768, 1), 0); del buf800  # reuse
        buf805 = buf799; del buf799  # reuse
        # Source Nodes: [l__mod___blocks_21_norm3, mul_86, mul_88, x_403, x_405, x_407], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf801, buf774, arg83_1, buf781, arg531_1, arg84_1, arg537_1, arg538_1, arg539_1, buf805, 6272, 768, grid=grid(6272), stream=stream0)
        del arg531_1
        del arg537_1
        del arg538_1
        del arg539_1
        del arg83_1
        del arg84_1
        del buf774
        del buf781
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        buf806 = extern_kernels.convolution(reinterpret_tensor(buf805, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg540_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf806, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg540_1
        buf807 = reinterpret_tensor(buf805, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf805  # reuse
        # Source Nodes: [x_409, x_410, x_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf806, arg541_1, arg700_1, arg701_1, arg542_1, arg543_1, buf807, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg541_1
        del arg542_1
        del arg543_1
        del arg700_1
        del arg701_1
        # Source Nodes: [x_409, x_410, x_411, x_412], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf808 = extern_kernels.convolution(buf807, arg544_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf808, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg544_1
        buf809 = buf739; del buf739  # reuse
        buf810 = buf738; del buf738  # reuse
        buf811 = buf737; del buf737  # reuse
        # Source Nodes: [l__mod___blocks_21_norm2, mul_89, x_414], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf801, arg86_1, buf808, arg545_1, buf809, buf810, buf811, 37632, 128, grid=grid(37632), stream=stream0)
        buf812 = buf741; del buf741  # reuse
        buf813 = buf740; del buf740  # reuse
        # Source Nodes: [l__mod___blocks_21_norm2, mul_89, x_414], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf809, buf810, buf811, buf812, buf813, 6272, 6, grid=grid(6272), stream=stream0)
        buf815 = reinterpret_tensor(buf807, (8, 784, 768), (602112, 768, 1), 0); del buf807  # reuse
        # Source Nodes: [l__mod___blocks_21_norm2, mul_89, x_414], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf801, arg86_1, buf808, arg545_1, buf812, buf813, arg546_1, arg547_1, buf815, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg546_1
        del arg547_1
        buf816 = reinterpret_tensor(buf780, (6272, 3072), (3072, 1), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf815, (6272, 768), (768, 1), 0), reinterpret_tensor(arg548_1, (768, 3072), (1, 768), 0), out=buf816)
        del arg548_1
        buf817 = reinterpret_tensor(buf816, (8, 784, 3072), (2408448, 3072, 1), 0); del buf816  # reuse
        # Source Nodes: [x_416], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf817, arg549_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg549_1
        buf818 = reinterpret_tensor(buf815, (6272, 768), (768, 1), 0); del buf815  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf817, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg550_1, (3072, 768), (1, 3072), 0), out=buf818)
        del arg550_1
        buf819 = reinterpret_tensor(buf818, (8, 784, 768), (602112, 768, 1), 0); del buf818  # reuse
        buf823 = reinterpret_tensor(buf806, (8, 784, 768), (602112, 768, 1), 0); del buf806  # reuse
        # Source Nodes: [l__mod___blocks_22_norm1, mul_89, mul_90, x_414, x_422], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_31.run(buf819, buf801, arg86_1, buf808, arg545_1, arg87_1, arg551_1, arg552_1, arg553_1, buf823, 6272, 768, grid=grid(6272), stream=stream0)
        del arg545_1
        del arg551_1
        del arg552_1
        del arg553_1
        del arg86_1
        del arg87_1
        del buf801
        buf824 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf823, (6272, 768), (768, 1), 0), reinterpret_tensor(arg554_1, (768, 2304), (1, 768), 0), out=buf824)
        del arg554_1
        buf825 = buf789; del buf789  # reuse
        # Source Nodes: [q_45], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf824, arg555_1, buf825, 43008, 112, grid=grid(43008), stream=stream0)
        buf826 = buf790; del buf790  # reuse
        # Source Nodes: [q_45], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf825, buf826, 6144, 7, grid=grid(6144), stream=stream0)
        buf827 = buf825; del buf825  # reuse
        # Source Nodes: [k_45], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf824, arg555_1, buf827, 43008, 112, grid=grid(43008), stream=stream0)
        buf828 = buf788; del buf788  # reuse
        # Source Nodes: [k_45], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf827, buf828, 6144, 7, grid=grid(6144), stream=stream0)
        buf829 = reinterpret_tensor(buf823, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf823  # reuse
        # Source Nodes: [matmul_44, q_45], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf824, arg555_1, buf826, buf829, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf830 = reinterpret_tensor(buf808, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf808  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf824, arg555_1, buf828, buf830, 4816896, grid=grid(4816896), stream=stream0)
        buf831 = reinterpret_tensor(buf796, (128, 48, 48), (2304, 48, 1), 0); del buf796  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf829, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf830, (128, 784, 48), (37632, 48, 1), 0), out=buf831)
        buf834 = reinterpret_tensor(buf793, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf793  # reuse
        # Source Nodes: [attn_66, attn_67], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf831, arg89_1, buf834, 6144, 48, grid=grid(6144), stream=stream0)
        del arg89_1
        buf835 = reinterpret_tensor(buf830, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf830  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf824, arg555_1, buf835, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg555_1
        buf836 = reinterpret_tensor(buf829, (128, 48, 784), (37632, 784, 1), 0); del buf829  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf834, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf835, (128, 48, 784), (37632, 784, 1), 0), out=buf836)
        buf837 = reinterpret_tensor(buf835, (8, 784, 768), (602112, 768, 1), 0); del buf835  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf836, buf837, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf838 = reinterpret_tensor(buf836, (6272, 768), (768, 1), 0); del buf836  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (6272, 768), (768, 1), 0), reinterpret_tensor(arg556_1, (768, 768), (1, 768), 0), out=buf838)
        del arg556_1
        buf842 = buf837; del buf837  # reuse
        # Source Nodes: [l__mod___blocks_22_norm3, mul_92, x_424, x_426], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf819, arg88_1, buf838, arg557_1, arg558_1, arg559_1, buf842, 6272, 768, grid=grid(6272), stream=stream0)
        del arg558_1
        del arg559_1
        # Source Nodes: [x_428], Original ATen: [aten.convolution]
        buf843 = extern_kernels.convolution(reinterpret_tensor(buf842, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg560_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf843, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg560_1
        buf844 = reinterpret_tensor(buf842, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf842  # reuse
        # Source Nodes: [x_428, x_429, x_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf843, arg561_1, arg703_1, arg704_1, arg562_1, arg563_1, buf844, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg561_1
        del arg562_1
        del arg563_1
        del arg703_1
        del arg704_1
        # Source Nodes: [x_428, x_429, x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf845 = extern_kernels.convolution(buf844, arg564_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf845, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg564_1
        buf846 = reinterpret_tensor(buf844, (8, 784, 768), (602112, 768, 1), 0); del buf844  # reuse
        buf850 = reinterpret_tensor(buf843, (8, 784, 768), (602112, 768, 1), 0); del buf843  # reuse
        # Source Nodes: [l__mod___blocks_22_norm2, mul_92, mul_93, x_424, x_426, x_433], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_33.run(buf819, arg88_1, buf838, arg557_1, arg90_1, buf845, arg565_1, arg566_1, arg567_1, buf846, buf850, 6272, 768, grid=grid(6272), stream=stream0)
        del arg557_1
        del arg565_1
        del arg566_1
        del arg567_1
        del arg88_1
        del arg90_1
        del buf819
        buf851 = reinterpret_tensor(buf817, (6272, 3072), (3072, 1), 0); del buf817  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf850, (6272, 768), (768, 1), 0), reinterpret_tensor(arg568_1, (768, 3072), (1, 768), 0), out=buf851)
        del arg568_1
        buf852 = reinterpret_tensor(buf851, (8, 784, 3072), (2408448, 3072, 1), 0); del buf851  # reuse
        # Source Nodes: [x_435], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf852, arg569_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg569_1
        buf853 = reinterpret_tensor(buf850, (6272, 768), (768, 1), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf852, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg570_1, (3072, 768), (1, 3072), 0), out=buf853)
        del arg570_1
        buf857 = reinterpret_tensor(buf845, (8, 784, 768), (602112, 768, 1), 0); del buf845  # reuse
        # Source Nodes: [l__mod___blocks_23_norm1, mul_94, x_441], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_32.run(buf846, arg91_1, buf853, arg571_1, arg572_1, arg573_1, buf857, 6272, 768, grid=grid(6272), stream=stream0)
        del arg572_1
        del arg573_1
        buf858 = buf824; del buf824  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf857, (6272, 768), (768, 1), 0), reinterpret_tensor(arg574_1, (768, 2304), (1, 768), 0), out=buf858)
        del arg574_1
        buf859 = buf827; del buf827  # reuse
        # Source Nodes: [q_47], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_13.run(buf858, arg575_1, buf859, 43008, 112, grid=grid(43008), stream=stream0)
        buf860 = buf828; del buf828  # reuse
        # Source Nodes: [q_47], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf859, buf860, 6144, 7, grid=grid(6144), stream=stream0)
        buf861 = buf859; del buf859  # reuse
        # Source Nodes: [k_47], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_15.run(buf858, arg575_1, buf861, 43008, 112, grid=grid(43008), stream=stream0)
        buf862 = buf826; del buf826  # reuse
        # Source Nodes: [k_47], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_14.run(buf861, buf862, 6144, 7, grid=grid(6144), stream=stream0)
        del buf861
        buf863 = reinterpret_tensor(buf857, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf857  # reuse
        # Source Nodes: [matmul_46, q_47], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_16.run(buf858, arg575_1, buf860, buf863, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del buf860
        buf864 = reinterpret_tensor(buf838, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf838  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf858, arg575_1, buf862, buf864, 4816896, grid=grid(4816896), stream=stream0)
        buf865 = reinterpret_tensor(buf834, (128, 48, 48), (2304, 48, 1), 0); del buf834  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf863, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf864, (128, 784, 48), (37632, 48, 1), 0), out=buf865)
        buf868 = reinterpret_tensor(buf831, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf831  # reuse
        # Source Nodes: [attn_69, attn_70], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf865, arg93_1, buf868, 6144, 48, grid=grid(6144), stream=stream0)
        del arg93_1
        del buf865
        buf869 = reinterpret_tensor(buf864, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf864  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf858, arg575_1, buf869, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg575_1
        del buf858
        buf870 = reinterpret_tensor(buf863, (128, 48, 784), (37632, 784, 1), 0); del buf863  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf868, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf869, (128, 48, 784), (37632, 784, 1), 0), out=buf870)
        del buf868
        buf871 = reinterpret_tensor(buf869, (8, 784, 768), (602112, 768, 1), 0); del buf869  # reuse
        # Source Nodes: [x_443], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf870, buf871, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf872 = reinterpret_tensor(buf870, (6272, 768), (768, 1), 0); del buf870  # reuse
        # Source Nodes: [x_443], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf871, (6272, 768), (768, 1), 0), reinterpret_tensor(arg576_1, (768, 768), (1, 768), 0), out=buf872)
        del arg576_1
        buf873 = reinterpret_tensor(buf872, (8, 784, 768), (602112, 768, 1), 0); del buf872  # reuse
        buf877 = buf871; del buf871  # reuse
        # Source Nodes: [l__mod___blocks_23_norm3, mul_94, mul_96, x_441, x_443, x_445], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_34.run(buf873, buf846, arg91_1, buf853, arg571_1, arg92_1, arg577_1, arg578_1, arg579_1, buf877, 6272, 768, grid=grid(6272), stream=stream0)
        del arg571_1
        del arg577_1
        del arg578_1
        del arg579_1
        del arg91_1
        del arg92_1
        del buf846
        del buf853
        # Source Nodes: [x_447], Original ATen: [aten.convolution]
        buf878 = extern_kernels.convolution(reinterpret_tensor(buf877, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg580_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf878, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg580_1
        buf879 = reinterpret_tensor(buf877, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf877  # reuse
        # Source Nodes: [x_447, x_448, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_24.run(buf878, arg581_1, arg706_1, arg707_1, arg582_1, arg583_1, buf879, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg581_1
        del arg582_1
        del arg583_1
        del arg706_1
        del arg707_1
        del buf878
        # Source Nodes: [x_447, x_448, x_449, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.gelu]
        buf880 = extern_kernels.convolution(buf879, arg584_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf880, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg584_1
        buf881 = buf811; del buf811  # reuse
        buf882 = buf810; del buf810  # reuse
        buf883 = buf809; del buf809  # reuse
        # Source Nodes: [l__mod___blocks_23_norm2, mul_97, x_452], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf873, arg94_1, buf880, arg585_1, buf881, buf882, buf883, 37632, 128, grid=grid(37632), stream=stream0)
        buf884 = buf813; del buf813  # reuse
        buf885 = buf812; del buf812  # reuse
        # Source Nodes: [l__mod___blocks_23_norm2, mul_97, x_452], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_22.run(buf881, buf882, buf883, buf884, buf885, 6272, 6, grid=grid(6272), stream=stream0)
        del buf881
        del buf882
        del buf883
        buf887 = reinterpret_tensor(buf879, (8, 784, 768), (602112, 768, 1), 0); del buf879  # reuse
        # Source Nodes: [l__mod___blocks_23_norm2, mul_97, x_452], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_poi_fused_add_mul_native_layer_norm_30.run(buf873, arg94_1, buf880, arg585_1, buf884, buf885, arg586_1, arg587_1, buf887, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del arg586_1
        del arg587_1
        del buf884
        del buf885
        buf888 = reinterpret_tensor(buf852, (6272, 3072), (3072, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf887, (6272, 768), (768, 1), 0), reinterpret_tensor(arg588_1, (768, 3072), (1, 768), 0), out=buf888)
        del arg588_1
        buf889 = reinterpret_tensor(buf888, (8, 784, 3072), (2408448, 3072, 1), 0); del buf888  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf889, arg589_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg589_1
        buf890 = reinterpret_tensor(buf887, (6272, 768), (768, 1), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf889, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg590_1, (3072, 768), (1, 3072), 0), out=buf890)
        del arg590_1
        del buf889
        buf891 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf895 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_10, x_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_35.run(arg96_1, buf873, arg94_1, buf880, arg585_1, arg95_1, buf890, arg591_1, arg592_1, arg593_1, buf891, buf895, 6280, 768, grid=grid(6280), stream=stream0)
        del arg585_1
        del arg591_1
        del arg592_1
        del arg593_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf873
        del buf880
        del buf890
        buf896 = reinterpret_tensor(buf862, (8, 768), (768, 1), 0); del buf862  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg595_1, reinterpret_tensor(buf895, (8, 768), (602880, 1), 0), reinterpret_tensor(arg594_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf896)
        del arg594_1
        del arg595_1
        buf897 = empty((6280, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_k], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg597_1, reinterpret_tensor(buf895, (6280, 768), (768, 1), 0), reinterpret_tensor(arg596_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf897)
        del arg596_1
        del arg597_1
        buf898 = empty((6280, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_v], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg599_1, reinterpret_tensor(buf895, (6280, 768), (768, 1), 0), reinterpret_tensor(arg598_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf898)
        del arg598_1
        del arg599_1
        # Source Nodes: [x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf899 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf896, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf897, (8, 16, 785, 48), (602880, 48, 768, 1), 0), reinterpret_tensor(buf898, (8, 16, 785, 48), (602880, 48, 768, 1), 0), None, False)
        buf900 = buf899[0]
        del buf899
        buf904 = buf896; del buf896  # reuse
        # Source Nodes: [x_cls_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg601_1, reinterpret_tensor(buf900, (8, 768), (768, 1), 0), reinterpret_tensor(arg600_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf904)
        del arg600_1
        del arg601_1
        buf908 = reinterpret_tensor(buf898, (8, 785, 768), (602880, 768, 1), 0); del buf898  # reuse
        # Source Nodes: [cat_9, mul_99, x_462, x_res], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_cat_mul_native_layer_norm_36.run(buf891, arg97_1, buf904, buf895, arg602_1, arg603_1, buf908, 6280, 768, grid=grid(6280), stream=stream0)
        del arg602_1
        del arg603_1
        del arg97_1
        buf909 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_464], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (8, 768), (602880, 1), 0), reinterpret_tensor(arg604_1, (768, 3072), (1, 768), 0), out=buf909)
        del arg604_1
        buf910 = reinterpret_tensor(buf909, (8, 1, 3072), (3072, 3072, 1), 0); del buf909  # reuse
        # Source Nodes: [x_464, x_465], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_37.run(buf910, arg605_1, 24576, grid=grid(24576), stream=stream0)
        del arg605_1
        buf911 = buf904; del buf904  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf910, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg606_1, (3072, 768), (1, 3072), 0), out=buf911)
        del arg606_1
        buf915 = buf895; del buf895  # reuse
        # Source Nodes: [cat_8, x_472, x_norm1_1], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_38.run(buf908, arg98_1, buf911, arg607_1, arg608_1, arg609_1, buf915, 6280, 768, grid=grid(6280), stream=stream0)
        del arg608_1
        del arg609_1
        buf916 = reinterpret_tensor(buf900, (8, 768), (768, 1), 0); del buf900  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg611_1, reinterpret_tensor(buf915, (8, 768), (602880, 1), 0), reinterpret_tensor(arg610_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf916)
        del arg610_1
        del arg611_1
        buf917 = reinterpret_tensor(buf891, (6280, 768), (768, 1), 0); del buf891  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_k], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg613_1, reinterpret_tensor(buf915, (6280, 768), (768, 1), 0), reinterpret_tensor(arg612_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf917)
        del arg612_1
        del arg613_1
        buf918 = buf897; del buf897  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_v], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg615_1, reinterpret_tensor(buf915, (6280, 768), (768, 1), 0), reinterpret_tensor(arg614_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf918)
        del arg614_1
        del arg615_1
        # Source Nodes: [x_cls_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf919 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf916, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf917, (8, 16, 785, 48), (602880, 48, 768, 1), 0), reinterpret_tensor(buf918, (8, 16, 785, 48), (602880, 48, 768, 1), 0), None, False)
        del buf917
        buf920 = buf919[0]
        del buf919
        buf924 = buf916; del buf916  # reuse
        # Source Nodes: [x_cls_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg617_1, reinterpret_tensor(buf920, (8, 768), (768, 1), 0), reinterpret_tensor(arg616_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf924)
        del arg616_1
        del arg617_1
        del buf920
        buf925 = buf915; del buf915  # reuse
        buf929 = reinterpret_tensor(buf918, (8, 785, 768), (602880, 768, 1), 0); del buf918  # reuse
        # Source Nodes: [cat_7, cat_8, mul_101, x_472, x_473, x_res_1], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_cat_mul_native_layer_norm_39.run(buf925, buf908, arg98_1, buf911, arg607_1, arg99_1, buf924, arg618_1, arg619_1, buf929, 6280, 768, grid=grid(6280), stream=stream0)
        del arg607_1
        del arg618_1
        del arg619_1
        del arg98_1
        del arg99_1
        del buf908
        del buf925
        buf930 = reinterpret_tensor(buf910, (8, 3072), (3072, 1), 0); del buf910  # reuse
        # Source Nodes: [x_475], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf929, (8, 768), (602880, 1), 0), reinterpret_tensor(arg620_1, (768, 3072), (1, 768), 0), out=buf930)
        del arg620_1
        buf931 = reinterpret_tensor(buf930, (8, 1, 3072), (3072, 3072, 1), 0); del buf930  # reuse
        # Source Nodes: [x_475, x_476], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_37.run(buf931, arg621_1, 24576, grid=grid(24576), stream=stream0)
        del arg621_1
        buf932 = buf924; del buf924  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf931, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg622_1, (3072, 768), (1, 3072), 0), out=buf932)
        del arg622_1
        del buf931
        buf933 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        buf934 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_6, x_483, x_485], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_40.run(buf929, arg100_1, buf932, arg623_1, buf933, buf934, 6280, 768, grid=grid(6280), stream=stream0)
        buf936 = buf911; del buf911  # reuse
        # Source Nodes: [x_487], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf929, arg100_1, buf932, arg623_1, buf933, buf934, arg624_1, arg625_1, buf936, 6144, grid=grid(6144), stream=stream0)
        del arg100_1
        del arg623_1
        del arg624_1
        del arg625_1
        del buf929
        del buf932
        del buf933
        del buf934
        buf937 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_487, x_488], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg627_1, buf936, reinterpret_tensor(arg626_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf937)
        del arg626_1
        del arg627_1
        return (buf937, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg631_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg634_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg637_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg640_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg643_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg646_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg649_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg652_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg655_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg658_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg661_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg664_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg667_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg670_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg673_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg676_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg679_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg682_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg685_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg688_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg691_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg694_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg697_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg700_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg703_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg706_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg709_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('xcit_large_24_p8_224', benchmark_compiled_module)
