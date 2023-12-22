
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


# kernel path: /tmp/torchinductor_youkaichao/zy/czyirduix6cyyuec7dfvcmlnzfoyrtbb4zm5cpjn54ggbwkfp337.py
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
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/c2/cc2tqc5iq5ehygh64creepkyhbyii6cb2twjuvmb3ukxozfzpoln.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhpuoexohknaof5652zxlynafz75sgbncmsuvip5qp3jkyqcfta.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1, x1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 => add_3, mul_4, mul_5, sub_1
# x1 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 8
    y1 = (yindex // 8)
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (8*x2) + (100352*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwmf25bxtprtgc4fqot6oyi6htr5xhwjwtc7eo65bbuxa4fbebp.py
# Source Nodes: [cat_63], Original ATen: [aten.cat]
# cat_63 => cat
triton_poi_fused_cat_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (8*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-100352) + y0 + (12544*x2) + (100352*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwntzv7enq6h3glpnhfhsylsphdoxmoeyetp7qv5wi7gtvvqaldq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1 => add_7, mul_10, mul_11, sub_3
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 8
    y1 = (yindex // 8)
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
    tl.store(out_ptr0 + (y0 + (8*x2) + (100352*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cske7c7yetpamckpk3lhfilgnc3wlcu4boosc7bd6wj7ybhgryeu.py
# Source Nodes: [cat_62, shortcut_1], Original ATen: [aten.add, aten.cat]
# cat_62 => cat_1
# shortcut_1 => add_10
triton_poi_fused_add_cat_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp29 = tl.load(in_out_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (8*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-100352) + y0 + (12544*x2) + (100352*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-8) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (16*y3)), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zjywfqhqspotymfri6dqzp2kc3egj7hlznxt5akfv4wui2f7ca.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1, x1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 => add_12, mul_16, mul_17, sub_5
# x1_2 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 24
    y1 = (yindex // 24)
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (24*x2) + (301056*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxbge2yv6lcs5xm6a7jldgdg4loumhx3k2lphbopdhdnr7ejcae.py
# Source Nodes: [cat_61], Original ATen: [aten.cat]
# cat_61 => cat_2
triton_poi_fused_cat_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 48
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (24*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 48, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-301056) + y0 + (12544*x2) + (301056*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-24) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-24) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-24) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-24) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbo2e2t3xchwokwqurym772sqfhrwuezo6kbownzyfjpoxmfmja.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_8 => add_16, mul_22, mul_23, sub_7
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 48
    y1 = (yindex // 48)
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
    tl.store(out_ptr0 + (y0 + (48*x2) + (150528*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zazfsiketrcpicwakr4em2xccnkm5nfpkngjnvwyrfj6rvxouv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 => add_18, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 12
    y1 = (yindex // 12)
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
    tl.store(out_ptr0 + (y0 + (12*x2) + (37632*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqbbdiu7l2bz56qbgeo6rg45xbnnga2cnov6t2atzdltwn4losw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_1 => add_22, mul_31, mul_32, sub_10
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 3136
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (50176*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3ab5mbyd5n6vv4msarz4cyackmqwd4gupkuliwaqv6lreqhtg4d.py
# Source Nodes: [cat_60, getattr_getattr_l__mod___blocks___1_____0___shortcut_3, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_60 => cat_3
# getattr_getattr_l__mod___blocks___1_____0___shortcut_3 => add_24, mul_34, mul_35, sub_11
# shortcut_2 => add_25
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 24
    x2 = xindex
    y1 = (yindex // 24)
    y3 = yindex
    tmp29 = tl.load(in_ptr6 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (12*x2) + (37632*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 24, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-37632) + x2 + (3136*y0) + (37632*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwivgm2gernjm7u4fceeuivzqv2gp5jib5mokkhtoxmvolxlrrkv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1, x1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 => add_27, mul_37, mul_38, sub_12
# x1_4 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 36
    y1 = (yindex // 36)
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (36*x2) + (112896*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrqb5qz67st7mwzqxz2viqniw6zp5a2ce2yh7tll7eiru4btjkk.py
# Source Nodes: [cat_59], Original ATen: [aten.cat]
# cat_59 => cat_4
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 72
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 36, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (36*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 72, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-112896) + y0 + (3136*x2) + (112896*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-36) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-36) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-36) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-36) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceulx3xk3qk2isrxerrbebzhvvskeeyxx3ytdkftzedd7k3gk2w2.py
# Source Nodes: [cat_58, shortcut_3], Original ATen: [aten.add, aten.cat]
# cat_58 => cat_5
# shortcut_3 => add_34
triton_poi_fused_add_cat_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp29 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (12*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 24, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-37632) + y0 + (3136*x2) + (37632*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-12) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-12) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-12) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-12) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calerw7u56ta7b25kxgi7i45zhktjk4m4h2nmevzijohudemn26k.py
# Source Nodes: [x_16, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_16 => add_40, mul_55, mul_56, sub_18
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 576
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
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_youkaichao/du/cdujec4kqyyorxxaakc53bp5j2xzemyr7bjyoukoouxfijmtyvlx.py
# Source Nodes: [x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se => mean
# x_se_1 => convolution_19
# x_se_2 => relu_9
triton_poi_fused_convolution_mean_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6v6qye6g63okzyrlpincl74w3bwy7qmsdo3xgqw743govabguw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_17, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => add_41, clamp_max, clamp_min, div
# x_17 => mul_57
# x_se => mean
# x_se_1 => convolution_19
# x_se_2 => relu_9
# x_se_3 => convolution_20
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hilyuszxrgzsx7lerwomtjx7w4evngamgwyiyifmjx4fz36cf7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 => add_43, mul_59, mul_60, sub_19
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 20
    y1 = (yindex // 20)
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
    tl.store(out_ptr0 + (y0 + (20*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugzs4lvzoo57ru4uzoqof3uszlt6bce7i7a77nrsdevd3dsium3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_1 => add_47, mul_65, mul_66, sub_21
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (18816*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xgeogtnrlmtahtsxicjlaegg4l5moo3y2zqdhbbt7odxhj6ihx.py
# Source Nodes: [cat_56, getattr_getattr_l__mod___blocks___3_____0___shortcut_3, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_56 => cat_7
# getattr_getattr_l__mod___blocks___3_____0___shortcut_3 => add_49, mul_68, mul_69, sub_22
# shortcut_4 => add_50
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
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
    tmp29 = tl.load(in_ptr6 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (20*x2) + (15680*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-15680) + x2 + (784*y0) + (15680*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-20) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-20) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-20) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-20) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/caia7uobgsbo46orlqog6r5m37ncopbr7iuvspk5uwdi72dtc6a6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1, x1_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 => add_52, mul_71, mul_72, sub_23
# x1_8 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 60
    y1 = (yindex // 60)
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
    tl.store(out_ptr0 + (y0 + (60*x2) + (47040*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpbmfxyrkde5sbgqgeclx5gkhc7joiontjguy4w4ohizygrro25.py
# Source Nodes: [cat_55, x_se_4], Original ATen: [aten.cat, aten.mean]
# cat_55 => cat_8
# x_se_4 => mean_1
triton_per_fused_cat_mean_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 960
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 120
    r2 = rindex
    x1 = (xindex // 120)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (60*r2) + (47040*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + r2 + (784*x0) + (47040*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-60) + x0, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-60) + x0, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-60) + x0, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-60) + x0, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = 784.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (r2 + (784*x3)), tmp29, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlfbnfv5fgl64lyl6ee7rbuykl4k27p6yjwdylinzjzssonye5n.py
# Source Nodes: [x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_4 => mean_1
# x_se_5 => convolution_27
# x_se_6 => relu_12
triton_poi_fused_convolution_mean_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/zn/cznwda5z2nohtvfshtthywo3xvedvgsvyrqfrru5th5quxcupwpk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_21, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_55, clamp_max_1, clamp_min_1, div_1
# x_21 => mul_76
# x_se_4 => mean_1
# x_se_5 => convolution_27
# x_se_6 => relu_12
# x_se_7 => convolution_28
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sdwgdwj3fe74ph5tohznasxfsigiwhwg4x336v3aiaesaskytj.py
# Source Nodes: [cat_54, shortcut_5], Original ATen: [aten.add, aten.cat]
# cat_54 => cat_9
# shortcut_5 => add_60
triton_poi_fused_add_cat_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 40
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
    tmp29 = tl.load(in_out_ptr0 + (x2 + (40*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (20*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-15680) + y0 + (784*x2) + (15680*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-20) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-20) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-20) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-20) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (40*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2mcr3ae4b7davfkzec2mgpjhqcoa54o6ucakm7uvw2tq7pmjmx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1, x1_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 => add_62, mul_84, mul_85, sub_27
# x1_10 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 120
    y1 = (yindex // 120)
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
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyr5zbnoua4ccrsw6yky53m4mln2cnjsu4tv7ou5l72s67jf4c2s.py
# Source Nodes: [cat_53], Original ATen: [aten.cat]
# cat_53 => cat_10
triton_poi_fused_cat_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 240
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 120, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (120*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 240, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-94080) + y0 + (784*x2) + (94080*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-120) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-120) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-120) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-120) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqysgs62orfqagpqe2gpn32ulzeoiikophqdu5byr7on3uvgbiyj.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_26 => add_66, mul_90, mul_91, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 240
    y1 = (yindex // 240)
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqjcpavpiuh74ulvbtexe5iubz6fitxutp5zpj7jnonqkdthn6e.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 => add_68, mul_93, mul_94, sub_30
triton_poi_fused__native_batch_norm_legit_no_training_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 40
    y1 = (yindex // 40)
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
    tl.store(out_ptr0 + (y0 + (40*x2) + (7840*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czefwsxntciitalgujlzujmntcppgbt3emnfxv3pcehdo3ian3y7.py
# Source Nodes: [cat_52, getattr_getattr_l__mod___blocks___5_____0___shortcut_3, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_52 => cat_11
# getattr_getattr_l__mod___blocks___5_____0___shortcut_3 => add_74, mul_102, mul_103, sub_33
# shortcut_6 => add_75
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 80
    x2 = xindex
    y1 = (yindex // 80)
    y3 = yindex
    tmp29 = tl.load(in_ptr6 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (40*x2) + (7840*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-7840) + x2 + (196*y0) + (7840*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-40) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-40) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-40) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-40) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrn72df4435eldmgih5ayt3ef4kapcfwns5t6rqddyzrwtdjht5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1, x1_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 => add_77, mul_105, mul_106, sub_34
# x1_12 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 100
    y1 = (yindex // 100)
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
    tl.store(out_ptr0 + (y0 + (100*x2) + (19600*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clohzchf5uquw2kas4rmdzfw5s443az7unseyxnqxosmkcnub6n3.py
# Source Nodes: [cat_51], Original ATen: [aten.cat]
# cat_51 => cat_12
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 200
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (100*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 200, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-19600) + y0 + (196*x2) + (19600*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-100) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-100) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-100) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-100) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (200*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdsmsjifsroquofgls22slz5yqvauvmkcl6nmscgmsfxege7v75.py
# Source Nodes: [cat_50, shortcut_7], Original ATen: [aten.add, aten.cat]
# cat_50 => cat_13
# shortcut_7 => add_84
triton_poi_fused_add_cat_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 80
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp29 = tl.load(in_out_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (40*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-7840) + y0 + (196*x2) + (7840*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-40) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-40) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-40) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-40) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (80*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czef3fsuerrdt3qpuytz2p47l6lygngi7rr5jf3usznimeo4sijs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1, x1_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 => add_86, mul_117, mul_118, sub_38
# x1_14 => relu_17
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 92
    y1 = (yindex // 92)
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
    tl.store(out_ptr0 + (y0 + (92*x2) + (18032*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqemruliatqugdldfihf72mjpuv4hoidewih3tav73mlxukaefsg.py
# Source Nodes: [cat_49], Original ATen: [aten.cat]
# cat_49 => cat_14
triton_poi_fused_cat_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 92, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (92*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 184, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-18032) + y0 + (196*x2) + (18032*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-92) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-92) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-92) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-92) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (184*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdri4fnkgfyizzppib2ixipvafxmhytrs4ixsthhfiybufqv5uuo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1, x1_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1 => add_104, mul_141, mul_142, sub_46
# x1_18 => relu_21
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 240
    y1 = (yindex // 240)
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c754xtcepzgplomuhhfqauccfjrmof5g5jmihbjpleketn3vrchp.py
# Source Nodes: [cat_45, x_se_8], Original ATen: [aten.cat, aten.mean]
# cat_45 => cat_18
# x_se_8 => mean_2
triton_per_fused_cat_mean_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_38', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (240*r2) + (47040*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + r2 + (196*x0) + (47040*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 196.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfjzye4pevtzlmdnaicplpx4yo64qsbrmjl57mrp5limey3rvr3.py
# Source Nodes: [x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_10 => relu_23
# x_se_8 => mean_2
# x_se_9 => convolution_52
triton_poi_fused_convolution_mean_relu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hsco7ekhiz2w2klj4435zg6bdwc5ndlzib5fet3apzdksumnkh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___se_gate, x_39, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___6_____3___se_gate => add_107, clamp_max_2, clamp_min_2, div_2
# x_39 => mul_146
# x_se_10 => relu_23
# x_se_11 => convolution_53
# x_se_8 => mean_2
# x_se_9 => convolution_52
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
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
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpqji52pmp4n3cld5eo6ixwhuh5z772fma7wazooml5pejzfwg4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 => add_109, mul_148, mul_149, sub_48
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (56*x2) + (10976*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cectc2ney5jileyslcbsa2exri6yp5omsm25i6r4dvpsg3auatca.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___6_____3___shortcut_1 => add_113, mul_154, mul_155, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7h3isniwvgo7ye7nerrh6wmerdeqvlnweln77lyipcnolciwe3.py
# Source Nodes: [cat_44, getattr_getattr_l__mod___blocks___6_____3___shortcut_3, shortcut_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_44 => cat_19
# getattr_getattr_l__mod___blocks___6_____3___shortcut_3 => add_115, mul_157, mul_158, sub_51
# shortcut_10 => add_116
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 112
    x2 = xindex
    y1 = (yindex // 112)
    y3 = yindex
    tmp29 = tl.load(in_ptr6 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (56*x2) + (10976*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 112, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10976) + x2 + (196*y0) + (10976*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-56) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-56) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-56) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-56) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggquud7ywnmdvf2xjmv6rchmbinjyudrov5r7ef65xiphc2nbce.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1, x1_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 => add_118, mul_160, mul_161, sub_52
# x1_20 => relu_24
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr2kvo7442gwzymhxbcvvai67k66bxdjf2l35y2bdoky7kj2r3u.py
# Source Nodes: [cat_43, x_se_12], Original ATen: [aten.cat, aten.mean]
# cat_43 => cat_20
# x_se_12 => mean_3
triton_per_fused_cat_mean_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 672
    r2 = rindex
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 336, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (336*r2) + (65856*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 672, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-65856) + r2 + (196*x0) + (65856*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 196.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kyeodpueumcrcze6lmmslx7hhkwle7zqwofvm6sadha6x4ge4r.py
# Source Nodes: [x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_12 => mean_3
# x_se_13 => convolution_60
# x_se_14 => relu_26
triton_poi_fused_convolution_mean_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
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


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qx7yu2vuh56j7qdjr5hqzegog3pdbzvbkqgculcujp6ou4bmoi.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___se_gate, x_43, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___6_____4___se_gate => add_121, clamp_max_3, clamp_min_3, div_3
# x_43 => mul_165
# x_se_12 => mean_3
# x_se_13 => convolution_60
# x_se_14 => relu_26
# x_se_15 => convolution_61
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
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
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3eiccr6eplrdgqlyc47qgn3hqjhxo2ey5m5aixkrs4kthr64bby.py
# Source Nodes: [cat_42, shortcut_11], Original ATen: [aten.add, aten.cat]
# cat_42 => cat_21
# shortcut_11 => add_126
triton_poi_fused_add_cat_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp29 = tl.load(in_out_ptr0 + (x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (56*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 112, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10976) + y0 + (196*x2) + (10976*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-56) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-56) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-56) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-56) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (112*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctk3h4vhmg7xauj6baottjkey7jveebp3vdxgvnkilzg35gn2oqm.py
# Source Nodes: [cat_41], Original ATen: [aten.cat]
# cat_41 => cat_22
triton_poi_fused_cat_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 672
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 336, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (336*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 672, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-65856) + y0 + (196*x2) + (65856*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-336) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-336) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-336) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-336) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwhi5p6qozkmpl4qqw3psbwy2u6m4kdfgejtlrj2qmca76h4scd.py
# Source Nodes: [x_48, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# x_48 => add_132, mul_179, mul_180, sub_58
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
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


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5mffe5o2h2xzaxvbxkqtzp6y6bnuo2zpaxr6624pgnfarqslho.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___se_gate, x_49, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___7_____0___se_gate => add_133, clamp_max_4, clamp_min_4, div_4
# x_49 => mul_181
# x_se_16 => mean_4
# x_se_17 => convolution_67
# x_se_18 => relu_29
# x_se_19 => convolution_68
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
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
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv253ar7kei2b4unu7gbrlwgrve4rna5b2fkfdx4oqd7mqb3o56.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 => add_135, mul_183, mul_184, sub_59
triton_poi_fused__native_batch_norm_legit_no_training_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 80
    y1 = (yindex // 80)
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (3920*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcbzsggfy4wuwnni4qlncjquww2qrmsduznlkanrvlb5gt2qh3y.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_1 => add_139, mul_189, mul_190, sub_61
triton_poi_fused__native_batch_norm_legit_no_training_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (112*x2) + (5488*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xq/cxq4shzehnooiseqmanrnssxivpejizaebb2vsukqx2e4t7r3b4w.py
# Source Nodes: [cat_40, getattr_getattr_l__mod___blocks___7_____0___shortcut_3, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
# cat_40 => cat_23
# getattr_getattr_l__mod___blocks___7_____0___shortcut_3 => add_141, mul_192, mul_193, sub_62
# shortcut_12 => add_142
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
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
    tmp29 = tl.load(in_ptr6 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (80*x2) + (3920*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-3920) + x2 + (49*y0) + (3920*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-80) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5m/c5m7mxcmdvb7koukwlcorninaxwuqymt2g6vpbjcfkl5d26n3rj5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1, x1_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 => add_144, mul_195, mul_196, sub_63
# x1_24 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 480
    y1 = (yindex // 480)
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
    tl.store(out_ptr0 + (y0 + (480*x2) + (23520*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kg3y4n7bqwtgziolndueakstrpfnviuosc7xhfrvtdp4raklfh.py
# Source Nodes: [cat_39], Original ATen: [aten.cat]
# cat_39 => cat_24
triton_poi_fused_cat_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 960
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 480, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (480*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 960, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-23520) + y0 + (49*x2) + (23520*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-480) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-480) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-480) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-480) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwafvjfc6kq6uqhuqxqmoiueeqsgg3sctlsmxtsefoqejjjpnxl.py
# Source Nodes: [cat_38, shortcut_13], Original ATen: [aten.add, aten.cat]
# cat_38 => cat_25
# shortcut_13 => add_151
triton_poi_fused_add_cat_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 160
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp29 = tl.load(in_out_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (80*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-3920) + y0 + (49*x2) + (3920*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-80) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-80) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-80) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-80) + x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp30 = tmp28 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (160*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdeosiwhocsxphxmc473gfiuttlsjfnn2xbewzk5hbko2uij2vyw.py
# Source Nodes: [cat_37, x_se_20], Original ATen: [aten.cat, aten.mean]
# cat_37 => cat_26
# x_se_20 => mean_5
triton_per_fused_cat_mean_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 480, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (480*r2) + (23520*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 960, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-23520) + r2 + (49*x0) + (23520*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 49.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp29, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dmw7pxw6poqcqlmkdjwxk2ccuyulkpnkonuxlr46evlzcjtcp4.py
# Source Nodes: [x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_20 => mean_5
# x_se_21 => convolution_79
# x_se_22 => relu_34
triton_poi_fused_convolution_mean_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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


# kernel path: /tmp/torchinductor_youkaichao/5c/c5c742flily4rvmybz6fxlqjjihjhtl42d3c4fventmk54mixzcf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___se_gate, x_56, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___8_____1___se_gate => add_156, clamp_max_5, clamp_min_5, div_5
# x_56 => mul_212
# x_se_20 => mean_5
# x_se_21 => convolution_79
# x_se_22 => relu_34
# x_se_23 => convolution_80
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_60', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/ccz2fbaqxoskyh6cll4ubpylckelliou6csnnj6jbzxajg2ltvqi.py
# Source Nodes: [x_67, x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_67 => add_182, mul_245, mul_246, sub_79
# x_72 => relu_40
# x_73 => mean_7
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_61', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
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
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzzck7vo3yasxq6nwzjkkvntjwn6ar2mc2mbc6zxazgqk5ngiuc.py
# Source Nodes: [x_67, x_72, x_73, x_76, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_67 => add_182, mul_245, mul_246, sub_79
# x_72 => relu_40
# x_73 => mean_7
# x_76 => convolution_94
# x_77 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1 = args
    args.clear()
    assert_size_stride(arg0_1, (960, ), (1, ))
    assert_size_stride(arg1_1, (960, ), (1, ))
    assert_size_stride(arg2_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg3_1, (1000, ), (1, ))
    assert_size_stride(arg4_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg8_1, (8, ), (1, ))
    assert_size_stride(arg9_1, (8, ), (1, ))
    assert_size_stride(arg10_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg11_1, (8, ), (1, ))
    assert_size_stride(arg12_1, (8, ), (1, ))
    assert_size_stride(arg13_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg14_1, (8, ), (1, ))
    assert_size_stride(arg15_1, (8, ), (1, ))
    assert_size_stride(arg16_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg17_1, (8, ), (1, ))
    assert_size_stride(arg18_1, (8, ), (1, ))
    assert_size_stride(arg19_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg26_1, (48, ), (1, ))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg29_1, (12, ), (1, ))
    assert_size_stride(arg30_1, (12, ), (1, ))
    assert_size_stride(arg31_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg32_1, (12, ), (1, ))
    assert_size_stride(arg33_1, (12, ), (1, ))
    assert_size_stride(arg34_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg35_1, (16, ), (1, ))
    assert_size_stride(arg36_1, (16, ), (1, ))
    assert_size_stride(arg37_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg38_1, (24, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg41_1, (36, ), (1, ))
    assert_size_stride(arg42_1, (36, ), (1, ))
    assert_size_stride(arg43_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg44_1, (36, ), (1, ))
    assert_size_stride(arg45_1, (36, ), (1, ))
    assert_size_stride(arg46_1, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg47_1, (12, ), (1, ))
    assert_size_stride(arg48_1, (12, ), (1, ))
    assert_size_stride(arg49_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg50_1, (12, ), (1, ))
    assert_size_stride(arg51_1, (12, ), (1, ))
    assert_size_stride(arg52_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg53_1, (36, ), (1, ))
    assert_size_stride(arg54_1, (36, ), (1, ))
    assert_size_stride(arg55_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg56_1, (36, ), (1, ))
    assert_size_stride(arg57_1, (36, ), (1, ))
    assert_size_stride(arg58_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg59_1, (72, ), (1, ))
    assert_size_stride(arg60_1, (72, ), (1, ))
    assert_size_stride(arg61_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg62_1, (20, ), (1, ))
    assert_size_stride(arg63_1, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg64_1, (72, ), (1, ))
    assert_size_stride(arg65_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg66_1, (20, ), (1, ))
    assert_size_stride(arg67_1, (20, ), (1, ))
    assert_size_stride(arg68_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg69_1, (20, ), (1, ))
    assert_size_stride(arg70_1, (20, ), (1, ))
    assert_size_stride(arg71_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg72_1, (24, ), (1, ))
    assert_size_stride(arg73_1, (24, ), (1, ))
    assert_size_stride(arg74_1, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg75_1, (40, ), (1, ))
    assert_size_stride(arg76_1, (40, ), (1, ))
    assert_size_stride(arg77_1, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg78_1, (60, ), (1, ))
    assert_size_stride(arg79_1, (60, ), (1, ))
    assert_size_stride(arg80_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg81_1, (60, ), (1, ))
    assert_size_stride(arg82_1, (60, ), (1, ))
    assert_size_stride(arg83_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg84_1, (32, ), (1, ))
    assert_size_stride(arg85_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg86_1, (120, ), (1, ))
    assert_size_stride(arg87_1, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg88_1, (20, ), (1, ))
    assert_size_stride(arg89_1, (20, ), (1, ))
    assert_size_stride(arg90_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg91_1, (20, ), (1, ))
    assert_size_stride(arg92_1, (20, ), (1, ))
    assert_size_stride(arg93_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg94_1, (120, ), (1, ))
    assert_size_stride(arg95_1, (120, ), (1, ))
    assert_size_stride(arg96_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (120, ), (1, ))
    assert_size_stride(arg98_1, (120, ), (1, ))
    assert_size_stride(arg99_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (240, ), (1, ))
    assert_size_stride(arg101_1, (240, ), (1, ))
    assert_size_stride(arg102_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg103_1, (40, ), (1, ))
    assert_size_stride(arg104_1, (40, ), (1, ))
    assert_size_stride(arg105_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (40, ), (1, ))
    assert_size_stride(arg107_1, (40, ), (1, ))
    assert_size_stride(arg108_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (40, ), (1, ))
    assert_size_stride(arg110_1, (40, ), (1, ))
    assert_size_stride(arg111_1, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg112_1, (80, ), (1, ))
    assert_size_stride(arg113_1, (80, ), (1, ))
    assert_size_stride(arg114_1, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg115_1, (100, ), (1, ))
    assert_size_stride(arg116_1, (100, ), (1, ))
    assert_size_stride(arg117_1, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (100, ), (1, ))
    assert_size_stride(arg119_1, (100, ), (1, ))
    assert_size_stride(arg120_1, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg121_1, (40, ), (1, ))
    assert_size_stride(arg122_1, (40, ), (1, ))
    assert_size_stride(arg123_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (40, ), (1, ))
    assert_size_stride(arg125_1, (40, ), (1, ))
    assert_size_stride(arg126_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg127_1, (92, ), (1, ))
    assert_size_stride(arg128_1, (92, ), (1, ))
    assert_size_stride(arg129_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg130_1, (92, ), (1, ))
    assert_size_stride(arg131_1, (92, ), (1, ))
    assert_size_stride(arg132_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg133_1, (40, ), (1, ))
    assert_size_stride(arg134_1, (40, ), (1, ))
    assert_size_stride(arg135_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (40, ), (1, ))
    assert_size_stride(arg137_1, (40, ), (1, ))
    assert_size_stride(arg138_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg139_1, (92, ), (1, ))
    assert_size_stride(arg140_1, (92, ), (1, ))
    assert_size_stride(arg141_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (92, ), (1, ))
    assert_size_stride(arg143_1, (92, ), (1, ))
    assert_size_stride(arg144_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg145_1, (40, ), (1, ))
    assert_size_stride(arg146_1, (40, ), (1, ))
    assert_size_stride(arg147_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (40, ), (1, ))
    assert_size_stride(arg149_1, (40, ), (1, ))
    assert_size_stride(arg150_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg151_1, (240, ), (1, ))
    assert_size_stride(arg152_1, (240, ), (1, ))
    assert_size_stride(arg153_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (240, ), (1, ))
    assert_size_stride(arg155_1, (240, ), (1, ))
    assert_size_stride(arg156_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg157_1, (120, ), (1, ))
    assert_size_stride(arg158_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg161_1, (56, ), (1, ))
    assert_size_stride(arg162_1, (56, ), (1, ))
    assert_size_stride(arg163_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg164_1, (56, ), (1, ))
    assert_size_stride(arg165_1, (56, ), (1, ))
    assert_size_stride(arg166_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (80, ), (1, ))
    assert_size_stride(arg168_1, (80, ), (1, ))
    assert_size_stride(arg169_1, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg170_1, (112, ), (1, ))
    assert_size_stride(arg171_1, (112, ), (1, ))
    assert_size_stride(arg172_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg173_1, (336, ), (1, ))
    assert_size_stride(arg174_1, (336, ), (1, ))
    assert_size_stride(arg175_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg176_1, (336, ), (1, ))
    assert_size_stride(arg177_1, (336, ), (1, ))
    assert_size_stride(arg178_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg179_1, (168, ), (1, ))
    assert_size_stride(arg180_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg181_1, (672, ), (1, ))
    assert_size_stride(arg182_1, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg183_1, (56, ), (1, ))
    assert_size_stride(arg184_1, (56, ), (1, ))
    assert_size_stride(arg185_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg186_1, (56, ), (1, ))
    assert_size_stride(arg187_1, (56, ), (1, ))
    assert_size_stride(arg188_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg189_1, (336, ), (1, ))
    assert_size_stride(arg190_1, (336, ), (1, ))
    assert_size_stride(arg191_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg192_1, (336, ), (1, ))
    assert_size_stride(arg193_1, (336, ), (1, ))
    assert_size_stride(arg194_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (672, ), (1, ))
    assert_size_stride(arg197_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg198_1, (168, ), (1, ))
    assert_size_stride(arg199_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg200_1, (672, ), (1, ))
    assert_size_stride(arg201_1, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg202_1, (80, ), (1, ))
    assert_size_stride(arg203_1, (80, ), (1, ))
    assert_size_stride(arg204_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg205_1, (80, ), (1, ))
    assert_size_stride(arg206_1, (80, ), (1, ))
    assert_size_stride(arg207_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg208_1, (112, ), (1, ))
    assert_size_stride(arg209_1, (112, ), (1, ))
    assert_size_stride(arg210_1, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg211_1, (160, ), (1, ))
    assert_size_stride(arg212_1, (160, ), (1, ))
    assert_size_stride(arg213_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg214_1, (480, ), (1, ))
    assert_size_stride(arg215_1, (480, ), (1, ))
    assert_size_stride(arg216_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (480, ), (1, ))
    assert_size_stride(arg218_1, (480, ), (1, ))
    assert_size_stride(arg219_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg220_1, (80, ), (1, ))
    assert_size_stride(arg221_1, (80, ), (1, ))
    assert_size_stride(arg222_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg223_1, (80, ), (1, ))
    assert_size_stride(arg224_1, (80, ), (1, ))
    assert_size_stride(arg225_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg226_1, (480, ), (1, ))
    assert_size_stride(arg227_1, (480, ), (1, ))
    assert_size_stride(arg228_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg229_1, (480, ), (1, ))
    assert_size_stride(arg230_1, (480, ), (1, ))
    assert_size_stride(arg231_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg232_1, (240, ), (1, ))
    assert_size_stride(arg233_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg234_1, (960, ), (1, ))
    assert_size_stride(arg235_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg236_1, (80, ), (1, ))
    assert_size_stride(arg237_1, (80, ), (1, ))
    assert_size_stride(arg238_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg239_1, (80, ), (1, ))
    assert_size_stride(arg240_1, (80, ), (1, ))
    assert_size_stride(arg241_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg242_1, (480, ), (1, ))
    assert_size_stride(arg243_1, (480, ), (1, ))
    assert_size_stride(arg244_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (480, ), (1, ))
    assert_size_stride(arg246_1, (480, ), (1, ))
    assert_size_stride(arg247_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg248_1, (80, ), (1, ))
    assert_size_stride(arg249_1, (80, ), (1, ))
    assert_size_stride(arg250_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg251_1, (80, ), (1, ))
    assert_size_stride(arg252_1, (80, ), (1, ))
    assert_size_stride(arg253_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg254_1, (480, ), (1, ))
    assert_size_stride(arg255_1, (480, ), (1, ))
    assert_size_stride(arg256_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg257_1, (480, ), (1, ))
    assert_size_stride(arg258_1, (480, ), (1, ))
    assert_size_stride(arg259_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg260_1, (240, ), (1, ))
    assert_size_stride(arg261_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg262_1, (960, ), (1, ))
    assert_size_stride(arg263_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg264_1, (80, ), (1, ))
    assert_size_stride(arg265_1, (80, ), (1, ))
    assert_size_stride(arg266_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg267_1, (80, ), (1, ))
    assert_size_stride(arg268_1, (80, ), (1, ))
    assert_size_stride(arg269_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg270_1, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg271_1, (1280, ), (1, ))
    assert_size_stride(arg272_1, (960, ), (1, ))
    assert_size_stride(arg273_1, (960, ), (1, ))
    assert_size_stride(arg274_1, (16, ), (1, ))
    assert_size_stride(arg275_1, (16, ), (1, ))
    assert_size_stride(arg276_1, (), ())
    assert_size_stride(arg277_1, (8, ), (1, ))
    assert_size_stride(arg278_1, (8, ), (1, ))
    assert_size_stride(arg279_1, (), ())
    assert_size_stride(arg280_1, (8, ), (1, ))
    assert_size_stride(arg281_1, (8, ), (1, ))
    assert_size_stride(arg282_1, (), ())
    assert_size_stride(arg283_1, (8, ), (1, ))
    assert_size_stride(arg284_1, (8, ), (1, ))
    assert_size_stride(arg285_1, (), ())
    assert_size_stride(arg286_1, (8, ), (1, ))
    assert_size_stride(arg287_1, (8, ), (1, ))
    assert_size_stride(arg288_1, (), ())
    assert_size_stride(arg289_1, (24, ), (1, ))
    assert_size_stride(arg290_1, (24, ), (1, ))
    assert_size_stride(arg291_1, (), ())
    assert_size_stride(arg292_1, (24, ), (1, ))
    assert_size_stride(arg293_1, (24, ), (1, ))
    assert_size_stride(arg294_1, (), ())
    assert_size_stride(arg295_1, (48, ), (1, ))
    assert_size_stride(arg296_1, (48, ), (1, ))
    assert_size_stride(arg297_1, (), ())
    assert_size_stride(arg298_1, (12, ), (1, ))
    assert_size_stride(arg299_1, (12, ), (1, ))
    assert_size_stride(arg300_1, (), ())
    assert_size_stride(arg301_1, (12, ), (1, ))
    assert_size_stride(arg302_1, (12, ), (1, ))
    assert_size_stride(arg303_1, (), ())
    assert_size_stride(arg304_1, (16, ), (1, ))
    assert_size_stride(arg305_1, (16, ), (1, ))
    assert_size_stride(arg306_1, (), ())
    assert_size_stride(arg307_1, (24, ), (1, ))
    assert_size_stride(arg308_1, (24, ), (1, ))
    assert_size_stride(arg309_1, (), ())
    assert_size_stride(arg310_1, (36, ), (1, ))
    assert_size_stride(arg311_1, (36, ), (1, ))
    assert_size_stride(arg312_1, (), ())
    assert_size_stride(arg313_1, (36, ), (1, ))
    assert_size_stride(arg314_1, (36, ), (1, ))
    assert_size_stride(arg315_1, (), ())
    assert_size_stride(arg316_1, (12, ), (1, ))
    assert_size_stride(arg317_1, (12, ), (1, ))
    assert_size_stride(arg318_1, (), ())
    assert_size_stride(arg319_1, (12, ), (1, ))
    assert_size_stride(arg320_1, (12, ), (1, ))
    assert_size_stride(arg321_1, (), ())
    assert_size_stride(arg322_1, (36, ), (1, ))
    assert_size_stride(arg323_1, (36, ), (1, ))
    assert_size_stride(arg324_1, (), ())
    assert_size_stride(arg325_1, (36, ), (1, ))
    assert_size_stride(arg326_1, (36, ), (1, ))
    assert_size_stride(arg327_1, (), ())
    assert_size_stride(arg328_1, (72, ), (1, ))
    assert_size_stride(arg329_1, (72, ), (1, ))
    assert_size_stride(arg330_1, (), ())
    assert_size_stride(arg331_1, (20, ), (1, ))
    assert_size_stride(arg332_1, (20, ), (1, ))
    assert_size_stride(arg333_1, (), ())
    assert_size_stride(arg334_1, (20, ), (1, ))
    assert_size_stride(arg335_1, (20, ), (1, ))
    assert_size_stride(arg336_1, (), ())
    assert_size_stride(arg337_1, (24, ), (1, ))
    assert_size_stride(arg338_1, (24, ), (1, ))
    assert_size_stride(arg339_1, (), ())
    assert_size_stride(arg340_1, (40, ), (1, ))
    assert_size_stride(arg341_1, (40, ), (1, ))
    assert_size_stride(arg342_1, (), ())
    assert_size_stride(arg343_1, (60, ), (1, ))
    assert_size_stride(arg344_1, (60, ), (1, ))
    assert_size_stride(arg345_1, (), ())
    assert_size_stride(arg346_1, (60, ), (1, ))
    assert_size_stride(arg347_1, (60, ), (1, ))
    assert_size_stride(arg348_1, (), ())
    assert_size_stride(arg349_1, (20, ), (1, ))
    assert_size_stride(arg350_1, (20, ), (1, ))
    assert_size_stride(arg351_1, (), ())
    assert_size_stride(arg352_1, (20, ), (1, ))
    assert_size_stride(arg353_1, (20, ), (1, ))
    assert_size_stride(arg354_1, (), ())
    assert_size_stride(arg355_1, (120, ), (1, ))
    assert_size_stride(arg356_1, (120, ), (1, ))
    assert_size_stride(arg357_1, (), ())
    assert_size_stride(arg358_1, (120, ), (1, ))
    assert_size_stride(arg359_1, (120, ), (1, ))
    assert_size_stride(arg360_1, (), ())
    assert_size_stride(arg361_1, (240, ), (1, ))
    assert_size_stride(arg362_1, (240, ), (1, ))
    assert_size_stride(arg363_1, (), ())
    assert_size_stride(arg364_1, (40, ), (1, ))
    assert_size_stride(arg365_1, (40, ), (1, ))
    assert_size_stride(arg366_1, (), ())
    assert_size_stride(arg367_1, (40, ), (1, ))
    assert_size_stride(arg368_1, (40, ), (1, ))
    assert_size_stride(arg369_1, (), ())
    assert_size_stride(arg370_1, (40, ), (1, ))
    assert_size_stride(arg371_1, (40, ), (1, ))
    assert_size_stride(arg372_1, (), ())
    assert_size_stride(arg373_1, (80, ), (1, ))
    assert_size_stride(arg374_1, (80, ), (1, ))
    assert_size_stride(arg375_1, (), ())
    assert_size_stride(arg376_1, (100, ), (1, ))
    assert_size_stride(arg377_1, (100, ), (1, ))
    assert_size_stride(arg378_1, (), ())
    assert_size_stride(arg379_1, (100, ), (1, ))
    assert_size_stride(arg380_1, (100, ), (1, ))
    assert_size_stride(arg381_1, (), ())
    assert_size_stride(arg382_1, (40, ), (1, ))
    assert_size_stride(arg383_1, (40, ), (1, ))
    assert_size_stride(arg384_1, (), ())
    assert_size_stride(arg385_1, (40, ), (1, ))
    assert_size_stride(arg386_1, (40, ), (1, ))
    assert_size_stride(arg387_1, (), ())
    assert_size_stride(arg388_1, (92, ), (1, ))
    assert_size_stride(arg389_1, (92, ), (1, ))
    assert_size_stride(arg390_1, (), ())
    assert_size_stride(arg391_1, (92, ), (1, ))
    assert_size_stride(arg392_1, (92, ), (1, ))
    assert_size_stride(arg393_1, (), ())
    assert_size_stride(arg394_1, (40, ), (1, ))
    assert_size_stride(arg395_1, (40, ), (1, ))
    assert_size_stride(arg396_1, (), ())
    assert_size_stride(arg397_1, (40, ), (1, ))
    assert_size_stride(arg398_1, (40, ), (1, ))
    assert_size_stride(arg399_1, (), ())
    assert_size_stride(arg400_1, (92, ), (1, ))
    assert_size_stride(arg401_1, (92, ), (1, ))
    assert_size_stride(arg402_1, (), ())
    assert_size_stride(arg403_1, (92, ), (1, ))
    assert_size_stride(arg404_1, (92, ), (1, ))
    assert_size_stride(arg405_1, (), ())
    assert_size_stride(arg406_1, (40, ), (1, ))
    assert_size_stride(arg407_1, (40, ), (1, ))
    assert_size_stride(arg408_1, (), ())
    assert_size_stride(arg409_1, (40, ), (1, ))
    assert_size_stride(arg410_1, (40, ), (1, ))
    assert_size_stride(arg411_1, (), ())
    assert_size_stride(arg412_1, (240, ), (1, ))
    assert_size_stride(arg413_1, (240, ), (1, ))
    assert_size_stride(arg414_1, (), ())
    assert_size_stride(arg415_1, (240, ), (1, ))
    assert_size_stride(arg416_1, (240, ), (1, ))
    assert_size_stride(arg417_1, (), ())
    assert_size_stride(arg418_1, (56, ), (1, ))
    assert_size_stride(arg419_1, (56, ), (1, ))
    assert_size_stride(arg420_1, (), ())
    assert_size_stride(arg421_1, (56, ), (1, ))
    assert_size_stride(arg422_1, (56, ), (1, ))
    assert_size_stride(arg423_1, (), ())
    assert_size_stride(arg424_1, (80, ), (1, ))
    assert_size_stride(arg425_1, (80, ), (1, ))
    assert_size_stride(arg426_1, (), ())
    assert_size_stride(arg427_1, (112, ), (1, ))
    assert_size_stride(arg428_1, (112, ), (1, ))
    assert_size_stride(arg429_1, (), ())
    assert_size_stride(arg430_1, (336, ), (1, ))
    assert_size_stride(arg431_1, (336, ), (1, ))
    assert_size_stride(arg432_1, (), ())
    assert_size_stride(arg433_1, (336, ), (1, ))
    assert_size_stride(arg434_1, (336, ), (1, ))
    assert_size_stride(arg435_1, (), ())
    assert_size_stride(arg436_1, (56, ), (1, ))
    assert_size_stride(arg437_1, (56, ), (1, ))
    assert_size_stride(arg438_1, (), ())
    assert_size_stride(arg439_1, (56, ), (1, ))
    assert_size_stride(arg440_1, (56, ), (1, ))
    assert_size_stride(arg441_1, (), ())
    assert_size_stride(arg442_1, (336, ), (1, ))
    assert_size_stride(arg443_1, (336, ), (1, ))
    assert_size_stride(arg444_1, (), ())
    assert_size_stride(arg445_1, (336, ), (1, ))
    assert_size_stride(arg446_1, (336, ), (1, ))
    assert_size_stride(arg447_1, (), ())
    assert_size_stride(arg448_1, (672, ), (1, ))
    assert_size_stride(arg449_1, (672, ), (1, ))
    assert_size_stride(arg450_1, (), ())
    assert_size_stride(arg451_1, (80, ), (1, ))
    assert_size_stride(arg452_1, (80, ), (1, ))
    assert_size_stride(arg453_1, (), ())
    assert_size_stride(arg454_1, (80, ), (1, ))
    assert_size_stride(arg455_1, (80, ), (1, ))
    assert_size_stride(arg456_1, (), ())
    assert_size_stride(arg457_1, (112, ), (1, ))
    assert_size_stride(arg458_1, (112, ), (1, ))
    assert_size_stride(arg459_1, (), ())
    assert_size_stride(arg460_1, (160, ), (1, ))
    assert_size_stride(arg461_1, (160, ), (1, ))
    assert_size_stride(arg462_1, (), ())
    assert_size_stride(arg463_1, (480, ), (1, ))
    assert_size_stride(arg464_1, (480, ), (1, ))
    assert_size_stride(arg465_1, (), ())
    assert_size_stride(arg466_1, (480, ), (1, ))
    assert_size_stride(arg467_1, (480, ), (1, ))
    assert_size_stride(arg468_1, (), ())
    assert_size_stride(arg469_1, (80, ), (1, ))
    assert_size_stride(arg470_1, (80, ), (1, ))
    assert_size_stride(arg471_1, (), ())
    assert_size_stride(arg472_1, (80, ), (1, ))
    assert_size_stride(arg473_1, (80, ), (1, ))
    assert_size_stride(arg474_1, (), ())
    assert_size_stride(arg475_1, (480, ), (1, ))
    assert_size_stride(arg476_1, (480, ), (1, ))
    assert_size_stride(arg477_1, (), ())
    assert_size_stride(arg478_1, (480, ), (1, ))
    assert_size_stride(arg479_1, (480, ), (1, ))
    assert_size_stride(arg480_1, (), ())
    assert_size_stride(arg481_1, (80, ), (1, ))
    assert_size_stride(arg482_1, (80, ), (1, ))
    assert_size_stride(arg483_1, (), ())
    assert_size_stride(arg484_1, (80, ), (1, ))
    assert_size_stride(arg485_1, (80, ), (1, ))
    assert_size_stride(arg486_1, (), ())
    assert_size_stride(arg487_1, (480, ), (1, ))
    assert_size_stride(arg488_1, (480, ), (1, ))
    assert_size_stride(arg489_1, (), ())
    assert_size_stride(arg490_1, (480, ), (1, ))
    assert_size_stride(arg491_1, (480, ), (1, ))
    assert_size_stride(arg492_1, (), ())
    assert_size_stride(arg493_1, (80, ), (1, ))
    assert_size_stride(arg494_1, (80, ), (1, ))
    assert_size_stride(arg495_1, (), ())
    assert_size_stride(arg496_1, (80, ), (1, ))
    assert_size_stride(arg497_1, (80, ), (1, ))
    assert_size_stride(arg498_1, (), ())
    assert_size_stride(arg499_1, (480, ), (1, ))
    assert_size_stride(arg500_1, (480, ), (1, ))
    assert_size_stride(arg501_1, (), ())
    assert_size_stride(arg502_1, (480, ), (1, ))
    assert_size_stride(arg503_1, (480, ), (1, ))
    assert_size_stride(arg504_1, (), ())
    assert_size_stride(arg505_1, (80, ), (1, ))
    assert_size_stride(arg506_1, (80, ), (1, ))
    assert_size_stride(arg507_1, (), ())
    assert_size_stride(arg508_1, (80, ), (1, ))
    assert_size_stride(arg509_1, (80, ), (1, ))
    assert_size_stride(arg510_1, (), ())
    assert_size_stride(arg511_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg511_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg511_1
        buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg4_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg4_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del buf1
        buf3 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg274_1, arg275_1, arg5_1, arg6_1, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg274_1
        del arg275_1
        del arg5_1
        del arg6_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg7_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del arg7_1
        buf5 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1, x1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf4, arg277_1, arg278_1, arg8_1, arg9_1, buf5, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg277_1
        del arg278_1
        del arg8_1
        del arg9_1
        del buf4
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg10_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf6, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del arg10_1
        buf7 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_4.run(buf5, buf6, arg280_1, arg281_1, arg11_1, arg12_1, buf7, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del arg11_1
        del arg12_1
        del arg280_1
        del arg281_1
        del buf5
        # Source Nodes: [cat_63, getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del arg13_1
        del buf7
        buf9 = reinterpret_tensor(buf6, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf6  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf8, arg283_1, arg284_1, arg14_1, arg15_1, buf9, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg14_1
        del arg15_1
        del arg283_1
        del arg284_1
        del buf8
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf10, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del arg16_1
        buf11 = buf3; del buf3  # reuse
        # Source Nodes: [cat_62, shortcut_1], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_6.run(buf11, buf9, buf10, arg286_1, arg287_1, arg17_1, arg18_1, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del arg17_1
        del arg18_1
        del arg286_1
        del arg287_1
        del buf10
        del buf9
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 112, 112), (301056, 12544, 112, 1))
        del arg19_1
        buf13 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1, x1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, arg289_1, arg290_1, arg20_1, arg21_1, buf13, 192, 12544, grid=grid(192, 12544), stream=stream0)
        del arg20_1
        del arg21_1
        del arg289_1
        del arg290_1
        del buf12
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg22_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf14, (8, 24, 112, 112), (301056, 12544, 112, 1))
        del arg22_1
        buf15 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf13, buf14, arg292_1, arg293_1, arg23_1, arg24_1, buf15, 100352, 48, grid=grid(100352, 48), stream=stream0)
        del arg23_1
        del arg24_1
        del arg292_1
        del arg293_1
        del buf13
        del buf14
        # Source Nodes: [cat_61, x_7], Original ATen: [aten.cat, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg25_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf16, (8, 48, 56, 56), (150528, 3136, 56, 1))
        del arg25_1
        del buf15
        buf17 = reinterpret_tensor(buf0, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf0  # reuse
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf16, arg295_1, arg296_1, arg26_1, arg27_1, buf17, 384, 3136, grid=grid(384, 3136), stream=stream0)
        del arg26_1
        del arg27_1
        del arg295_1
        del arg296_1
        del buf16
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 12, 56, 56), (37632, 3136, 56, 1))
        del arg28_1
        del buf17
        buf19 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf18, arg298_1, arg299_1, arg29_1, arg30_1, buf19, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del arg298_1
        del arg299_1
        del arg29_1
        del arg30_1
        del buf18
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf20, (8, 12, 56, 56), (37632, 3136, 56, 1))
        del arg31_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf11, arg34_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf21, (8, 16, 56, 56), (50176, 3136, 56, 1))
        del arg34_1
        del buf11
        buf22 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf21, arg304_1, arg305_1, arg35_1, arg36_1, buf22, 128, 3136, grid=grid(128, 3136), stream=stream0)
        del arg304_1
        del arg305_1
        del arg35_1
        del arg36_1
        del buf21
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1, getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg37_1
        del buf22
        buf24 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_60, getattr_getattr_l__mod___blocks___1_____0___shortcut_3, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12.run(buf19, buf20, arg301_1, arg302_1, arg32_1, arg33_1, buf23, arg307_1, arg308_1, arg38_1, arg39_1, buf24, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg301_1
        del arg302_1
        del arg307_1
        del arg308_1
        del arg32_1
        del arg33_1
        del arg38_1
        del arg39_1
        del buf19
        del buf23
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 36, 56, 56), (112896, 3136, 56, 1))
        del arg40_1
        buf26 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1, x1_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf25, arg310_1, arg311_1, arg41_1, arg42_1, buf26, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg310_1
        del arg311_1
        del arg41_1
        del arg42_1
        del buf25
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf27, (8, 36, 56, 56), (112896, 3136, 56, 1))
        del arg43_1
        buf28 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf26, buf27, arg313_1, arg314_1, arg44_1, arg45_1, buf28, 25088, 72, grid=grid(25088, 72), stream=stream0)
        del arg313_1
        del arg314_1
        del arg44_1
        del arg45_1
        del buf26
        # Source Nodes: [cat_59, getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 12, 56, 56), (37632, 3136, 56, 1))
        del arg46_1
        buf30 = reinterpret_tensor(buf20, (8, 12, 56, 56), (37632, 1, 672, 12), 0); del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf29, arg316_1, arg317_1, arg47_1, arg48_1, buf30, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del arg316_1
        del arg317_1
        del arg47_1
        del arg48_1
        del buf29
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf31, (8, 12, 56, 56), (37632, 3136, 56, 1))
        del arg49_1
        buf32 = buf24; del buf24  # reuse
        # Source Nodes: [cat_58, shortcut_3], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_15.run(buf32, buf30, buf31, arg319_1, arg320_1, arg50_1, arg51_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg319_1
        del arg320_1
        del arg50_1
        del arg51_1
        del buf30
        del buf31
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 36, 56, 56), (112896, 3136, 56, 1))
        del arg52_1
        buf34 = reinterpret_tensor(buf27, (8, 36, 56, 56), (112896, 1, 2016, 36), 0); del buf27  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1, x1_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf33, arg322_1, arg323_1, arg53_1, arg54_1, buf34, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg322_1
        del arg323_1
        del arg53_1
        del arg54_1
        del buf33
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg55_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf35, (8, 36, 56, 56), (112896, 3136, 56, 1))
        del arg55_1
        buf36 = buf28; del buf28  # reuse
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf34, buf35, arg325_1, arg326_1, arg56_1, arg57_1, buf36, 25088, 72, grid=grid(25088, 72), stream=stream0)
        del arg325_1
        del arg326_1
        del arg56_1
        del arg57_1
        del buf34
        del buf35
        # Source Nodes: [cat_57, x_15], Original ATen: [aten.cat, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg58_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf37, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg58_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf40 = reinterpret_tensor(buf39, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf39  # reuse
        # Source Nodes: [x_16, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_16.run(buf38, buf40, arg328_1, arg329_1, arg59_1, arg60_1, 576, 784, grid=grid(576), stream=stream0)
        del arg328_1
        del arg329_1
        del arg59_1
        del arg60_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf41 = extern_kernels.convolution(buf40, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg61_1
        del buf40
        buf42 = reinterpret_tensor(buf41, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf41  # reuse
        # Source Nodes: [x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_17.run(buf42, arg62_1, 160, grid=grid(160), stream=stream0)
        del arg62_1
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 72, 1, 1), (72, 1, 1, 1))
        del arg63_1
        del buf42
        buf44 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_17, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_18.run(buf38, buf43, arg64_1, buf44, 576, 784, grid=grid(576, 784), stream=stream0)
        del arg64_1
        del buf38
        del buf43
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___3_____0___se_gate, x_17, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf45 = extern_kernels.convolution(buf44, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 20, 28, 28), (15680, 784, 28, 1))
        del arg65_1
        del buf44
        buf46 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf45, arg331_1, arg332_1, arg66_1, arg67_1, buf46, 160, 784, grid=grid(160, 784), stream=stream0)
        del arg331_1
        del arg332_1
        del arg66_1
        del arg67_1
        del buf45
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg68_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf47, (8, 20, 28, 28), (15680, 784, 28, 1))
        del arg68_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf32, arg71_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf48, (8, 24, 28, 28), (18816, 784, 28, 1))
        del arg71_1
        del buf32
        buf49 = empty_strided((8, 24, 28, 28), (18816, 1, 672, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf48, arg337_1, arg338_1, arg72_1, arg73_1, buf49, 192, 784, grid=grid(192, 784), stream=stream0)
        del arg337_1
        del arg338_1
        del arg72_1
        del arg73_1
        del buf48
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1, getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg74_1
        del buf49
        buf51 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_56, getattr_getattr_l__mod___blocks___3_____0___shortcut_3, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_21.run(buf46, buf47, arg334_1, arg335_1, arg69_1, arg70_1, buf50, arg340_1, arg341_1, arg75_1, arg76_1, buf51, 320, 784, grid=grid(320, 784), stream=stream0)
        del arg334_1
        del arg335_1
        del arg340_1
        del arg341_1
        del arg69_1
        del arg70_1
        del arg75_1
        del arg76_1
        del buf46
        del buf50
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg77_1
        buf53 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1, x1_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf52, arg343_1, arg344_1, arg78_1, arg79_1, buf53, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg343_1
        del arg344_1
        del arg78_1
        del arg79_1
        del buf52
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf54, (8, 60, 28, 28), (47040, 784, 28, 1))
        del arg80_1
        buf55 = empty((8, 120, 28, 28), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf57 = reinterpret_tensor(buf56, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf56  # reuse
        # Source Nodes: [cat_55, x_se_4], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_23.run(buf57, buf53, buf54, arg346_1, arg347_1, arg81_1, arg82_1, buf55, 960, 784, grid=grid(960), stream=stream0)
        del arg346_1
        del arg347_1
        del arg81_1
        del arg82_1
        del buf53
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf58 = extern_kernels.convolution(buf57, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg83_1
        del buf57
        buf59 = reinterpret_tensor(buf58, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf58  # reuse
        # Source Nodes: [x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf59, arg84_1, 256, grid=grid(256), stream=stream0)
        del arg84_1
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg85_1
        del buf59
        buf61 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_21, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_25.run(buf55, buf60, arg86_1, buf61, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg86_1
        del buf55
        del buf60
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___4_____0___se_gate, x_21, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 20, 28, 28), (15680, 784, 28, 1))
        del arg87_1
        buf63 = reinterpret_tensor(buf47, (8, 20, 28, 28), (15680, 1, 560, 20), 0); del buf47  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf62, arg349_1, arg350_1, arg88_1, arg89_1, buf63, 160, 784, grid=grid(160, 784), stream=stream0)
        del arg349_1
        del arg350_1
        del arg88_1
        del arg89_1
        del buf62
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf64, (8, 20, 28, 28), (15680, 784, 28, 1))
        del arg90_1
        buf65 = buf51; del buf51  # reuse
        # Source Nodes: [cat_54, shortcut_5], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_26.run(buf65, buf63, buf64, arg352_1, arg353_1, arg91_1, arg92_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg352_1
        del arg353_1
        del arg91_1
        del arg92_1
        del buf63
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg93_1
        buf67 = buf61; del buf61  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1, x1_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf66, arg355_1, arg356_1, arg94_1, arg95_1, buf67, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg355_1
        del arg356_1
        del arg94_1
        del arg95_1
        del buf66
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf68, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg96_1
        buf69 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_28.run(buf67, buf68, arg358_1, arg359_1, arg97_1, arg98_1, buf69, 6272, 240, grid=grid(6272, 240), stream=stream0)
        del arg358_1
        del arg359_1
        del arg97_1
        del arg98_1
        # Source Nodes: [cat_53, x_25], Original ATen: [aten.cat, aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg99_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf70, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg99_1
        del buf69
        buf71 = reinterpret_tensor(buf54, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf54  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf70, arg361_1, arg362_1, arg100_1, arg101_1, buf71, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg100_1
        del arg101_1
        del arg361_1
        del arg362_1
        del buf70
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg102_1
        buf73 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf72, arg364_1, arg365_1, arg103_1, arg104_1, buf73, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg103_1
        del arg104_1
        del arg364_1
        del arg365_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf74, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg105_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf65, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf75, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg108_1
        del buf65
        buf76 = reinterpret_tensor(buf72, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf72  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf75, arg370_1, arg371_1, arg109_1, arg110_1, buf76, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg109_1
        del arg110_1
        del arg370_1
        del arg371_1
        del buf75
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1, getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg111_1
        del buf76
        buf78 = reinterpret_tensor(buf64, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf64  # reuse
        # Source Nodes: [cat_52, getattr_getattr_l__mod___blocks___5_____0___shortcut_3, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_31.run(buf73, buf74, arg367_1, arg368_1, arg106_1, arg107_1, buf77, arg373_1, arg374_1, arg112_1, arg113_1, buf78, 640, 196, grid=grid(640, 196), stream=stream0)
        del arg106_1
        del arg107_1
        del arg112_1
        del arg113_1
        del arg367_1
        del arg368_1
        del arg373_1
        del arg374_1
        del buf73
        del buf77
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 100, 14, 14), (19600, 196, 14, 1))
        del arg114_1
        buf80 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1, x1_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf79, arg376_1, arg377_1, arg115_1, arg116_1, buf80, 800, 196, grid=grid(800, 196), stream=stream0)
        del arg115_1
        del arg116_1
        del arg376_1
        del arg377_1
        del buf79
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=100, bias=None)
        assert_size_stride(buf81, (8, 100, 14, 14), (19600, 196, 14, 1))
        del arg117_1
        buf82 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf80, buf81, arg379_1, arg380_1, arg118_1, arg119_1, buf82, 1568, 200, grid=grid(1568, 200), stream=stream0)
        del arg118_1
        del arg119_1
        del arg379_1
        del arg380_1
        del buf80
        del buf81
        # Source Nodes: [cat_51, getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg120_1
        del buf82
        buf84 = reinterpret_tensor(buf74, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf74  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf83, arg382_1, arg383_1, arg121_1, arg122_1, buf84, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg121_1
        del arg122_1
        del arg382_1
        del arg383_1
        del buf83
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf85, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg123_1
        buf86 = buf78; del buf78  # reuse
        # Source Nodes: [cat_50, shortcut_7], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_34.run(buf86, buf84, buf85, arg385_1, arg386_1, arg124_1, arg125_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg124_1
        del arg125_1
        del arg385_1
        del arg386_1
        del buf84
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 92, 14, 14), (18032, 196, 14, 1))
        del arg126_1
        buf88 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1, x1_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf87, arg388_1, arg389_1, arg127_1, arg128_1, buf88, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg127_1
        del arg128_1
        del arg388_1
        del arg389_1
        del buf87
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg129_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf89, (8, 92, 14, 14), (18032, 196, 14, 1))
        del arg129_1
        buf90 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_36.run(buf88, buf89, arg391_1, arg392_1, arg130_1, arg131_1, buf90, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del arg130_1
        del arg131_1
        del arg391_1
        del arg392_1
        del buf88
        # Source Nodes: [cat_49, getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg132_1
        buf92 = reinterpret_tensor(buf85, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf85  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf91, arg394_1, arg395_1, arg133_1, arg134_1, buf92, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg133_1
        del arg134_1
        del arg394_1
        del arg395_1
        del buf91
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf93, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg135_1
        buf94 = buf86; del buf86  # reuse
        # Source Nodes: [cat_48, shortcut_8], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_34.run(buf94, buf92, buf93, arg397_1, arg398_1, arg136_1, arg137_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg136_1
        del arg137_1
        del arg397_1
        del arg398_1
        del buf92
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 92, 14, 14), (18032, 196, 14, 1))
        del arg138_1
        buf96 = reinterpret_tensor(buf89, (8, 92, 14, 14), (18032, 1, 1288, 92), 0); del buf89  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1, x1_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf95, arg400_1, arg401_1, arg139_1, arg140_1, buf96, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg139_1
        del arg140_1
        del arg400_1
        del arg401_1
        del buf95
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf97, (8, 92, 14, 14), (18032, 196, 14, 1))
        del arg141_1
        buf98 = buf90; del buf90  # reuse
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_36.run(buf96, buf97, arg403_1, arg404_1, arg142_1, arg143_1, buf98, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del arg142_1
        del arg143_1
        del arg403_1
        del arg404_1
        del buf96
        del buf97
        # Source Nodes: [cat_47, getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg144_1
        del buf98
        buf100 = reinterpret_tensor(buf93, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf93  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_30.run(buf99, arg406_1, arg407_1, arg145_1, arg146_1, buf100, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg145_1
        del arg146_1
        del arg406_1
        del arg407_1
        del buf99
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf101, (8, 40, 14, 14), (7840, 196, 14, 1))
        del arg147_1
        buf102 = buf94; del buf94  # reuse
        # Source Nodes: [cat_46, shortcut_9], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_34.run(buf102, buf100, buf101, arg409_1, arg410_1, arg148_1, arg149_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg148_1
        del arg149_1
        del arg409_1
        del arg410_1
        del buf100
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg150_1
        buf104 = buf71; del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1, x1_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf103, arg412_1, arg413_1, arg151_1, arg152_1, buf104, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg151_1
        del arg152_1
        del arg412_1
        del arg413_1
        del buf103
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf105, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg153_1
        buf106 = reinterpret_tensor(buf68, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf68  # reuse
        buf107 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf108 = reinterpret_tensor(buf107, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf107  # reuse
        # Source Nodes: [cat_45, x_se_8], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_38.run(buf108, buf104, buf105, arg415_1, arg416_1, arg154_1, arg155_1, buf106, 3840, 196, grid=grid(3840), stream=stream0)
        del arg154_1
        del arg155_1
        del arg415_1
        del arg416_1
        # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
        buf109 = extern_kernels.convolution(buf108, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg156_1
        del buf108
        buf110 = reinterpret_tensor(buf109, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf109  # reuse
        # Source Nodes: [x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_39.run(buf110, arg157_1, 960, grid=grid(960), stream=stream0)
        del arg157_1
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg158_1
        del buf110
        buf112 = reinterpret_tensor(buf67, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf67  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___se_gate, x_39, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_40.run(buf106, buf111, arg159_1, buf112, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg159_1
        del buf106
        del buf111
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___6_____3___se_gate, x_39, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 56, 14, 14), (10976, 196, 14, 1))
        del arg160_1
        del buf112
        buf114 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf113, arg418_1, arg419_1, arg161_1, arg162_1, buf114, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg161_1
        del arg162_1
        del arg418_1
        del arg419_1
        del buf113
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg163_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf115, (8, 56, 14, 14), (10976, 196, 14, 1))
        del arg163_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_0], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf102, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf116, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg166_1
        buf117 = buf102; del buf102  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf116, arg424_1, arg425_1, arg167_1, arg168_1, buf117, 640, 196, grid=grid(640, 196), stream=stream0)
        del arg167_1
        del arg168_1
        del arg424_1
        del arg425_1
        del buf116
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1, getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg169_1
        del buf117
        buf119 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_44, getattr_getattr_l__mod___blocks___6_____3___shortcut_3, shortcut_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_43.run(buf114, buf115, arg421_1, arg422_1, arg164_1, arg165_1, buf118, arg427_1, arg428_1, arg170_1, arg171_1, buf119, 896, 196, grid=grid(896, 196), stream=stream0)
        del arg164_1
        del arg165_1
        del arg170_1
        del arg171_1
        del arg421_1
        del arg422_1
        del arg427_1
        del arg428_1
        del buf114
        del buf118
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg172_1
        buf121 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1, x1_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf120, arg430_1, arg431_1, arg173_1, arg174_1, buf121, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg173_1
        del arg174_1
        del arg430_1
        del arg431_1
        del buf120
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg175_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf122, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg175_1
        buf123 = empty((8, 672, 14, 14), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf125 = reinterpret_tensor(buf124, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf124  # reuse
        # Source Nodes: [cat_43, x_se_12], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_45.run(buf125, buf121, buf122, arg433_1, arg434_1, arg176_1, arg177_1, buf123, 5376, 196, grid=grid(5376), stream=stream0)
        del arg176_1
        del arg177_1
        del arg433_1
        del arg434_1
        del buf121
        # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
        buf126 = extern_kernels.convolution(buf125, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg178_1
        del buf125
        buf127 = reinterpret_tensor(buf126, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf126  # reuse
        # Source Nodes: [x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_46.run(buf127, arg179_1, 1344, grid=grid(1344), stream=stream0)
        del arg179_1
        # Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf128 = extern_kernels.convolution(buf127, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg180_1
        del buf127
        buf129 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___se_gate, x_43, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_47.run(buf123, buf128, arg181_1, buf129, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg181_1
        del buf123
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___6_____4___se_gate, x_43, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf130 = extern_kernels.convolution(buf129, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 56, 14, 14), (10976, 196, 14, 1))
        del arg182_1
        buf131 = reinterpret_tensor(buf115, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf115  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf130, arg436_1, arg437_1, arg183_1, arg184_1, buf131, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg183_1
        del arg184_1
        del arg436_1
        del arg437_1
        del buf130
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg185_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf132, (8, 56, 14, 14), (10976, 196, 14, 1))
        del arg185_1
        buf133 = buf119; del buf119  # reuse
        # Source Nodes: [cat_42, shortcut_11], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_48.run(buf133, buf131, buf132, arg439_1, arg440_1, arg186_1, arg187_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg186_1
        del arg187_1
        del arg439_1
        del arg440_1
        del buf131
        del buf132
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg188_1
        buf135 = reinterpret_tensor(buf122, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf122  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1, x1_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf134, arg442_1, arg443_1, arg189_1, arg190_1, buf135, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg189_1
        del arg190_1
        del arg442_1
        del arg443_1
        del buf134
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf136, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg191_1
        buf137 = buf129; del buf129  # reuse
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf135, buf136, arg445_1, arg446_1, arg192_1, arg193_1, buf137, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del arg192_1
        del arg193_1
        del arg445_1
        del arg446_1
        del buf135
        del buf136
        # Source Nodes: [cat_41, x_47], Original ATen: [aten.cat, aten.convolution]
        buf138 = extern_kernels.convolution(buf137, arg194_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf138, (8, 672, 7, 7), (32928, 49, 7, 1))
        del arg194_1
        del buf137
        buf139 = buf138; del buf138  # reuse
        buf140 = reinterpret_tensor(buf128, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf128  # reuse
        buf141 = reinterpret_tensor(buf140, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf140  # reuse
        # Source Nodes: [x_48, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_50.run(buf139, buf141, arg448_1, arg449_1, arg195_1, arg196_1, 5376, 49, grid=grid(5376), stream=stream0)
        del arg195_1
        del arg196_1
        del arg448_1
        del arg449_1
        # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
        buf142 = extern_kernels.convolution(buf141, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg197_1
        del buf141
        buf143 = reinterpret_tensor(buf142, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf142  # reuse
        # Source Nodes: [x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_46.run(buf143, arg198_1, 1344, grid=grid(1344), stream=stream0)
        del arg198_1
        # Source Nodes: [x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf144 = extern_kernels.convolution(buf143, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg199_1
        del buf143
        buf145 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___se_gate, x_49, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_51.run(buf139, buf144, arg200_1, buf145, 5376, 49, grid=grid(5376, 49), stream=stream0)
        del arg200_1
        del buf139
        del buf144
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___7_____0___se_gate, x_49, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf146 = extern_kernels.convolution(buf145, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg201_1
        del buf145
        buf147 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf146, arg451_1, arg452_1, arg202_1, arg203_1, buf147, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg202_1
        del arg203_1
        del arg451_1
        del arg452_1
        del buf146
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf148, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg204_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf133, arg207_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf149, (8, 112, 7, 7), (5488, 49, 7, 1))
        del arg207_1
        del buf133
        buf150 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf149, arg457_1, arg458_1, arg208_1, arg209_1, buf150, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg208_1
        del arg209_1
        del arg457_1
        del arg458_1
        del buf149
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1, getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf151 = extern_kernels.convolution(buf150, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg210_1
        del buf150
        buf152 = reinterpret_tensor(buf101, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf101  # reuse
        # Source Nodes: [cat_40, getattr_getattr_l__mod___blocks___7_____0___shortcut_3, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_54.run(buf147, buf148, arg454_1, arg455_1, arg205_1, arg206_1, buf151, arg460_1, arg461_1, arg211_1, arg212_1, buf152, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del arg205_1
        del arg206_1
        del arg211_1
        del arg212_1
        del arg454_1
        del arg455_1
        del arg460_1
        del arg461_1
        del buf147
        del buf151
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg213_1
        buf154 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1, x1_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf153, arg463_1, arg464_1, arg214_1, arg215_1, buf154, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg214_1
        del arg215_1
        del arg463_1
        del arg464_1
        del buf153
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf155, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg216_1
        buf156 = reinterpret_tensor(buf105, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf105  # reuse
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf154, buf155, arg466_1, arg467_1, arg217_1, arg218_1, buf156, 392, 960, grid=grid(392, 960), stream=stream0)
        del arg217_1
        del arg218_1
        del arg466_1
        del arg467_1
        del buf154
        # Source Nodes: [cat_39, getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg219_1
        buf158 = reinterpret_tensor(buf148, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf148  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf157, arg469_1, arg470_1, arg220_1, arg221_1, buf158, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg220_1
        del arg221_1
        del arg469_1
        del arg470_1
        del buf157
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg222_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf159, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg222_1
        buf160 = buf152; del buf152  # reuse
        # Source Nodes: [cat_38, shortcut_13], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf160, buf158, buf159, arg472_1, arg473_1, arg223_1, arg224_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg223_1
        del arg224_1
        del arg472_1
        del arg473_1
        del buf158
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg225_1
        buf162 = reinterpret_tensor(buf155, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf155  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1, x1_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf161, arg475_1, arg476_1, arg226_1, arg227_1, buf162, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg226_1
        del arg227_1
        del arg475_1
        del arg476_1
        del buf161
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf163, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg228_1
        buf164 = reinterpret_tensor(buf156, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf156  # reuse
        buf165 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf165, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf165  # reuse
        # Source Nodes: [cat_37, x_se_20], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_58.run(buf166, buf162, buf163, arg478_1, arg479_1, arg229_1, arg230_1, buf164, 7680, 49, grid=grid(7680), stream=stream0)
        del arg229_1
        del arg230_1
        del arg478_1
        del arg479_1
        del buf162
        # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
        buf167 = extern_kernels.convolution(buf166, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg231_1
        del buf166
        buf168 = reinterpret_tensor(buf167, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf167  # reuse
        # Source Nodes: [x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_59.run(buf168, arg232_1, 1920, grid=grid(1920), stream=stream0)
        del arg232_1
        # Source Nodes: [x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf169 = extern_kernels.convolution(buf168, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg233_1
        del buf168
        buf170 = reinterpret_tensor(buf104, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf104  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___se_gate, x_56, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_60.run(buf164, buf169, arg234_1, buf170, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg234_1
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___8_____1___se_gate, x_56, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf171 = extern_kernels.convolution(buf170, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg235_1
        buf172 = reinterpret_tensor(buf159, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf159  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf171, arg481_1, arg482_1, arg236_1, arg237_1, buf172, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg236_1
        del arg237_1
        del arg481_1
        del arg482_1
        del buf171
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg238_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf173, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg238_1
        buf174 = buf160; del buf160  # reuse
        # Source Nodes: [cat_36, shortcut_14], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf174, buf172, buf173, arg484_1, arg485_1, arg239_1, arg240_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg239_1
        del arg240_1
        del arg484_1
        del arg485_1
        del buf172
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg241_1
        buf176 = reinterpret_tensor(buf163, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf163  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1, x1_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf175, arg487_1, arg488_1, arg242_1, arg243_1, buf176, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg242_1
        del arg243_1
        del arg487_1
        del arg488_1
        del buf175
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf177, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg244_1
        buf178 = buf170; del buf170  # reuse
        # Source Nodes: [cat_35], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf176, buf177, arg490_1, arg491_1, arg245_1, arg246_1, buf178, 392, 960, grid=grid(392, 960), stream=stream0)
        del arg245_1
        del arg246_1
        del arg490_1
        del arg491_1
        del buf176
        # Source Nodes: [cat_35, getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg247_1
        buf180 = reinterpret_tensor(buf173, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf173  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf179, arg493_1, arg494_1, arg248_1, arg249_1, buf180, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg248_1
        del arg249_1
        del arg493_1
        del arg494_1
        del buf179
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg250_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf181, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg250_1
        buf182 = buf174; del buf174  # reuse
        # Source Nodes: [cat_34, shortcut_15], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf182, buf180, buf181, arg496_1, arg497_1, arg251_1, arg252_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg251_1
        del arg252_1
        del arg496_1
        del arg497_1
        del buf180
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg253_1
        buf184 = reinterpret_tensor(buf177, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf177  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1, x1_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf183, arg499_1, arg500_1, arg254_1, arg255_1, buf184, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg254_1
        del arg255_1
        del arg499_1
        del arg500_1
        del buf183
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg256_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf185, (8, 480, 7, 7), (23520, 49, 7, 1))
        del arg256_1
        buf186 = reinterpret_tensor(buf178, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf178  # reuse
        buf187 = reinterpret_tensor(buf169, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf169  # reuse
        buf188 = reinterpret_tensor(buf187, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf187  # reuse
        # Source Nodes: [cat_33, x_se_24], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_58.run(buf188, buf184, buf185, arg502_1, arg503_1, arg257_1, arg258_1, buf186, 7680, 49, grid=grid(7680), stream=stream0)
        del arg257_1
        del arg258_1
        del arg502_1
        del arg503_1
        del buf184
        del buf185
        # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
        buf189 = extern_kernels.convolution(buf188, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg259_1
        del buf188
        buf190 = reinterpret_tensor(buf189, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf189  # reuse
        # Source Nodes: [x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_59.run(buf190, arg260_1, 1920, grid=grid(1920), stream=stream0)
        del arg260_1
        # Source Nodes: [x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf191 = extern_kernels.convolution(buf190, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg261_1
        del buf190
        buf192 = reinterpret_tensor(buf164, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf164  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___se_gate, x_63, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_60.run(buf186, buf191, arg262_1, buf192, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg262_1
        del buf186
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___8_____3___se_gate, x_63, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf193 = extern_kernels.convolution(buf192, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg263_1
        del buf192
        buf194 = reinterpret_tensor(buf181, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf181  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf193, arg505_1, arg506_1, arg264_1, arg265_1, buf194, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg264_1
        del arg265_1
        del arg505_1
        del arg506_1
        del buf193
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg266_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf195, (8, 80, 7, 7), (3920, 49, 7, 1))
        del arg266_1
        buf196 = buf182; del buf182  # reuse
        # Source Nodes: [cat_32, shortcut_16], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf196, buf194, buf195, arg508_1, arg509_1, arg267_1, arg268_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg267_1
        del arg268_1
        del arg508_1
        del arg509_1
        del buf194
        del buf195
        # Source Nodes: [cat_32, shortcut_16, x_66], Original ATen: [aten.add, aten.cat, aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg269_1
        del buf196
        buf198 = reinterpret_tensor(buf191, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf191  # reuse
        buf199 = reinterpret_tensor(buf198, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf198  # reuse
        # Source Nodes: [x_67, x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_61.run(buf199, buf197, arg272_1, arg273_1, arg0_1, arg1_1, 7680, 49, grid=grid(7680), stream=stream0)
        del arg0_1
        del arg1_1
        del arg272_1
        del arg273_1
        del buf197
        # Source Nodes: [x_67, x_72, x_73, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf200 = extern_kernels.convolution(buf199, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg270_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [x_67, x_72, x_73, x_76, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62.run(buf201, arg271_1, 10240, grid=grid(10240), stream=stream0)
        del arg271_1
        buf202 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg3_1, reinterpret_tensor(buf201, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg2_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf202)
        del arg2_1
        del arg3_1
        return (buf202, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg277_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg280_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg283_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg286_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg289_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg292_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg295_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg298_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg301_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg304_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg307_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg310_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg313_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg316_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg319_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg322_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg325_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg328_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg331_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg334_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg337_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg340_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg343_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg346_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg349_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg352_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg355_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg358_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg361_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg364_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg367_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg370_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg373_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg376_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg379_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg382_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg385_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg388_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg391_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg394_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg397_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg400_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg403_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg406_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg409_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg412_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg415_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg418_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg421_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg424_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg427_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg430_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg433_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg436_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg439_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg442_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg445_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg448_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg451_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg454_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg457_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg460_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg463_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg466_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg469_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg472_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg475_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg478_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg481_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg484_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg487_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg490_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg493_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg496_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg499_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg502_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg505_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg508_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg511_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
