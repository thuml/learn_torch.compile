
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tbfk7bem5bndzjfq4vqzbeig6bu6bi5eijif7wdgwjiu4ugt4c.py
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
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 109561
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
    tmp0 = tl.load(in_ptr0 + (x2 + (109561*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (328683*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/pp/cppsdlvqjjznqcqm522dmkacjdhpkgnjuk6svm3glwfqdfccl4hs.py
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
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
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


# kernel path: /tmp/torchinductor_youkaichao/ha/cha7yb64cwnrgpmzau3ua7trrqsag62nll3t5ogilf5grd6tb62x.py
# Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_5 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 27225
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (27225*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (27225*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (96*x2) + (2613600*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cihde7plarol3p3epd7f3snfjcy6pggjhw3ksktot4nfpdpzax3l.py
# Source Nodes: [l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool, l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool, l__mod___cell_stem_1_conv_prev_1x1_path_2_pad, max_pool2d, x_21, x_89], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.max_pool2d_with_indices, aten.relu]
# l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool => avg_pool2d
# l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool => avg_pool2d_1
# l__mod___cell_stem_1_conv_prev_1x1_path_2_pad => constant_pad_nd_9
# max_pool2d => max_pool2d_with_indices
# x_21 => constant_pad_nd_1
# x_89 => relu_14
triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 83)
    x2 = xindex % 83
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp76 = tl.load(in_ptr0 + ((2*x2) + (330*x3) + (27225*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-166) + (2*x2) + (330*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-165) + (2*x2) + (330*x3) + (27225*y4)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-164) + (2*x2) + (330*x3) + (27225*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x2) + (330*x3) + (27225*y4)), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x2) + (330*x3) + (27225*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x2) + (330*x3) + (27225*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x3)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (164 + (2*x2) + (330*x3) + (27225*y4)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (165 + (2*x2) + (330*x3) + (27225*y4)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (166 + (2*x2) + (330*x3) + (27225*y4)), tmp71 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp77 = triton_helpers.maximum(0, tmp76)
    tmp78 = 1.0
    tmp79 = tmp77 * tmp78
    tmp80 = triton_helpers.maximum(0, tmp72)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp71, tmp80, tmp81)
    tmp83 = tmp82 * tmp78
    tl.store(out_ptr0 + (y0 + (96*x5) + (661344*y1)), tmp75, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (96*x5) + (661344*y1)), tmp79, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (96*x5) + (661344*y1)), tmp83, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzf2b76aonycwgnui4frv5u7dfyp2ojg3kgs2tbf47wp6dt4q5z.py
# Source Nodes: [x_84, x_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_84 => relu_13
# x_right => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 27225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (27225*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (27225*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (54*x2) + (1470150*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqjfofjahckpixt56xblqwit4me3exd34qgfqcscmbhtglln6wjg.py
# Source Nodes: [x_22, x_24], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_22 => relu_3
# x_24 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 29241
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 171)
    x2 = xindex % 171
    y4 = yindex
    x5 = xindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = (-3) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-498) + x2 + (165*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (54*x5) + (1579014*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4xnlzul46f4v567iqykjsxdhxafehmk7evow3nvbci7bgcdhgm.py
# Source Nodes: [x_27], Original ATen: [aten.convolution]
# x_27 => convolution_8
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = tl.load(in_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (54*x2) + (372006*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmm46owchuj55xr7ddjiol5hy4hjn6idhvfp2yudauswqq64gmxx.py
# Source Nodes: [x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_28 => add_12, mul_16, mul_17, sub_5
# x_29 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = tl.load(in_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (54*x2) + (372006*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf35jpumcxcishpi3o37lzfezfa6veh4xcotqjgsyakj6xephmll.py
# Source Nodes: [x_36, x_38], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_36 => relu_5
# x_38 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 28561
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 169)
    x2 = xindex % 169
    y4 = yindex
    x5 = xindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-332) + x2 + (165*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (54*x5) + (1542294*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqylwbaejrqxbik72utw7adla2n2nb5dwxkahb3x2vevszb76a3b.py
# Source Nodes: [x_48, x_50], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_48 => relu_7
# x_50 => constant_pad_nd_5
triton_poi_fused_constant_pad_nd_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 27889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 167)
    x2 = xindex % 167
    y4 = yindex
    x5 = xindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-166) + x2 + (165*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (54*x5) + (1506006*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vsqxiqu4hrg3teff33rewaq36rdhkyd2ofnrnhfxtpdgsggcef.py
# Source Nodes: [x_60, x_comb_iter_2, x_comb_iter_2_left, x_comb_iter_2_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_60 => relu_9
# x_comb_iter_2 => add_24
# x_comb_iter_2_left => add_19, mul_25, mul_26, sub_8
# x_comb_iter_2_right => add_23, mul_31, mul_32, sub_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 432
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 54
    y1 = (yindex // 54)
    tmp0 = tl.load(in_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2 + (6889*y0) + (1860030*y1)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (54*x2) + (372006*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkzf4cghfofr74nolyu7hdnseajm2ta6ezoutzyjf6s4gxpcvd5.py
# Source Nodes: [x_35, x_71, x_comb_iter_1, x_comb_iter_1_left, x_comb_iter_1_right, x_comb_iter_3, x_comb_iter_3_left, x_comb_iter_3_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_35 => constant_pad_nd_3
# x_71 => constant_pad_nd_6
# x_comb_iter_1 => add_15
# x_comb_iter_1_left => add_14, mul_19, mul_20, sub_6
# x_comb_iter_1_right => max_pool2d_with_indices_1
# x_comb_iter_3 => add_29
# x_comb_iter_3_left => add_28, mul_37, mul_38, sub_12
# x_comb_iter_3_right => max_pool2d_with_indices_2
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 83) % 83
    x0 = xindex % 83
    x2 = (xindex // 6889)
    x4 = xindex
    x6 = (xindex // 6889) % 54
    x7 = (xindex // 372006)
    x8 = xindex % 372006
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x4), xmask)
    tmp93 = tl.load(in_ptr7 + (x6), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr8 + (x6), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr9 + (x6), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr10 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-166) + (2*x0) + (330*x1) + (27225*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-165) + (2*x0) + (330*x1) + (27225*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-164) + (2*x0) + (330*x1) + (27225*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (330*x1) + (27225*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (330*x1) + (27225*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (330*x1) + (27225*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (164 + (2*x0) + (330*x1) + (27225*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (165 + (2*x0) + (330*x1) + (27225*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (166 + (2*x0) + (330*x1) + (27225*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tmp94 = tmp92 - tmp93
    tmp96 = tmp95 + tmp80
    tmp97 = tl.sqrt(tmp96)
    tmp98 = 1 / tmp97
    tmp99 = tmp98 * tmp84
    tmp100 = tmp94 * tmp99
    tmp102 = tmp100 * tmp101
    tmp104 = tmp102 + tmp103
    tmp105 = tmp104 + tmp75
    tl.store(out_ptr2 + (x8 + (1860030*x7)), tmp91, xmask)
    tl.store(out_ptr3 + (x8 + (1860030*x7)), tmp105, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyytrvyeazhuzyn2ijorveie2vano42dg4yqflos6c6qgfxoll2a.py
# Source Nodes: [cat_34, x_left], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat]
# cat_34 => cat_1
# x_left => add_38, mul_49, mul_50, sub_16
triton_poi_fused__native_batch_norm_legit_no_training_cat_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5952096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 6889) % 108
    x2 = (xindex // 744012)
    x3 = xindex % 744012
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 54, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (372006*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 108, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-372006) + x3 + (372006*x2)), tmp8 & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3x5wpw6oqeq2zraokksgkfcz7tbccfihzaou2t3wvd7esfpsgj.py
# Source Nodes: [x_93, x_95], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_93 => relu_16
# x_95 => constant_pad_nd_10
triton_poi_fused_constant_pad_nd_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 7569
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 87)
    x2 = xindex % 87
    y4 = yindex
    x5 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-168) + x2 + (83*x3) + (6889*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (108*x5) + (817452*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltmsdwbw442udopjt6gh54wdaije3irutwksvps4dt6t3bf3koy.py
# Source Nodes: [x_98], Original ATen: [aten.convolution]
# x_98 => convolution_32
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (108*x2) + (190512*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkptwsarpmxaa67ghujyhpp4azhjo5ft4dqx3h4faskppalugjl.py
# Source Nodes: [x_100, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_100 => relu_17
# x_99 => add_42, mul_55, mul_56, sub_18
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (108*x2) + (190512*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c66wfdtjwfz7wnyawxoz6guvxjrfcb7cyhcoi5kzbg7lndn5lyvm.py
# Source Nodes: [x_106, x_comb_iter_0_left_1, x_comb_iter_0_right_1, x_comb_iter_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_106 => constant_pad_nd_11
# x_comb_iter_0_left_1 => add_44, mul_58, mul_59, sub_19
# x_comb_iter_0_right_1 => max_pool2d_with_indices_3
# x_comb_iter_5 => add_45
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex
    x6 = (xindex // 1764) % 108
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + (2*x0) + (166*x1) + (6889*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-83) + (2*x0) + (166*x1) + (6889*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-82) + (2*x0) + (166*x1) + (6889*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (166*x1) + (6889*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (166*x1) + (6889*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (166*x1) + (6889*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (82 + (2*x0) + (166*x1) + (6889*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (83 + (2*x0) + (166*x1) + (6889*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (84 + (2*x0) + (166*x1) + (6889*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tl.store(out_ptr1 + (x8 + (952560*x7)), tmp91, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjoutmpmkrgxp3m6fcjdsbgvirluzjyc7cpeuoeznn5v3p4kiguk.py
# Source Nodes: [x_10, x_8], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_10 => constant_pad_nd
# x_8 => relu_1
triton_poi_fused_constant_pad_nd_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 28561
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 169)
    x2 = xindex % 169
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-332) + x2 + (165*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (96*x5) + (2741856*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iqifodcf2245a5pylxjvhnyvmzxqr3wxxckslgqhiazquzah7g.py
# Source Nodes: [x_13], Original ATen: [aten.convolution]
# x_13 => convolution_3
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 6889
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
    tmp0 = tl.load(in_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (661344*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c25ayct46s2sxj24moagcpw4momqqmil4a4zu4i2qiuszjoiubum.py
# Source Nodes: [x_comb_iter_0, x_comb_iter_0_left, x_comb_iter_0_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_0 => add_10
# x_comb_iter_0_left => add_7, mul_10, mul_11, sub_3
# x_comb_iter_0_right => add_9, mul_13, mul_14, sub_4
triton_poi_fused__native_batch_norm_legit_no_training_add_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 6889) % 54
    x2 = (xindex // 372006)
    x4 = xindex % 372006
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x4 + (1860030*x2)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxogvkhkjicstcqfqr2jlf4zgaoo4ifa62n7v7vdl3k6kjpbsxey.py
# Source Nodes: [x_72, x_74], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_72 => relu_11
# x_74 => constant_pad_nd_7
triton_poi_fused_constant_pad_nd_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 27889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 167)
    x2 = xindex % 167
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-166) + x2 + (165*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (96*x5) + (2677344*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xq67tzu6se6d2ivdetz6rewliu7u4k4j5hg33477jibnyv32u5.py
# Source Nodes: [x_comb_iter_4, x_comb_iter_4_left, x_comb_iter_4_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_4 => add_36
# x_comb_iter_4_left => add_33, mul_43, mul_44, sub_14
# x_comb_iter_4_right => add_35, mul_46, mul_47, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 6889) % 54
    x2 = (xindex // 372006)
    x4 = xindex % 372006
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x4 + (1860030*x2)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckp7le5qnnvqk7lzrjfmcl27xiiwfsowg7g5idqk53hs6nsg3ieh.py
# Source Nodes: [x_90], Original ATen: [aten.relu]
# x_90 => relu_15
triton_poi_fused_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 270
    y1 = (yindex // 270)
    tmp0 = tl.load(in_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (270*x2) + (1860030*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ax/cax2kiwyffauy27356unae3t5cqi2jydh22gecl2wol4gans4zom.py
# Source Nodes: [x_169, x_right_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_169 => relu_28
# x_right_1 => add_40, mul_52, mul_53, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (6889*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (6889*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (108*x2) + (744012*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2m4ttnadye6avcrq7pf6raythcb7uv7gyljmpuxhwwmfoecf7o.py
# Source Nodes: [x_133, x_135], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_133 => relu_22
# x_135 => constant_pad_nd_15
triton_poi_fused_constant_pad_nd_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 7225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 85)
    x2 = xindex % 85
    y4 = yindex
    x5 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + x2 + (83*x3) + (6889*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (108*x5) + (780300*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmr56leh6mmnmzz5t34obhdkrzp32tzgqwn44e2bqfanryrdjcf5.py
# Source Nodes: [x_145, x_comb_iter_2_left_1, x_comb_iter_2_right_1, x_comb_iter_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_145 => relu_24
# x_comb_iter_2_left_1 => add_54, mul_70, mul_71, sub_23
# x_comb_iter_2_right_1 => add_58, mul_76, mul_77, sub_25
# x_comb_iter_7 => add_59
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2 + (1764*y0) + (952560*y1)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (108*x2) + (190512*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2rgyorpib2kxckytinehloy376d6qv2bqavd6el65v5wgmof2l.py
# Source Nodes: [x_107, x_109], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_107 => relu_18
# x_109 => constant_pad_nd_12
triton_poi_fused_constant_pad_nd_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 7921
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 89)
    x2 = xindex % 89
    y4 = yindex
    x5 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = (-3) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-252) + x2 + (83*x3) + (6889*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (108*x5) + (855468*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmaoadvd376zfw3g2oh5ou3hu2mc3pbehojo3mtd7edmlsgcp6j.py
# Source Nodes: [x_120, x_156, x_comb_iter_1_left_1, x_comb_iter_1_right_1, x_comb_iter_3_left_1, x_comb_iter_3_right_1, x_comb_iter_6, x_comb_iter_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_120 => constant_pad_nd_13
# x_156 => constant_pad_nd_16
# x_comb_iter_1_left_1 => add_49, mul_64, mul_65, sub_21
# x_comb_iter_1_right_1 => max_pool2d_with_indices_4
# x_comb_iter_3_left_1 => add_63, mul_82, mul_83, sub_27
# x_comb_iter_3_right_1 => max_pool2d_with_indices_5
# x_comb_iter_6 => add_50
# x_comb_iter_8 => add_64
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex
    x6 = (xindex // 1764) % 108
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x4), xmask)
    tmp93 = tl.load(in_ptr7 + (x6), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr8 + (x6), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr9 + (x6), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr10 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + (2*x0) + (166*x1) + (6889*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-83) + (2*x0) + (166*x1) + (6889*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-82) + (2*x0) + (166*x1) + (6889*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (166*x1) + (6889*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (166*x1) + (6889*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (166*x1) + (6889*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (82 + (2*x0) + (166*x1) + (6889*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (83 + (2*x0) + (166*x1) + (6889*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (84 + (2*x0) + (166*x1) + (6889*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tmp94 = tmp92 - tmp93
    tmp96 = tmp95 + tmp80
    tmp97 = tl.sqrt(tmp96)
    tmp98 = 1 / tmp97
    tmp99 = tmp98 * tmp84
    tmp100 = tmp94 * tmp99
    tmp102 = tmp100 * tmp101
    tmp104 = tmp102 + tmp103
    tmp105 = tmp104 + tmp75
    tl.store(out_ptr2 + (x8 + (952560*x7)), tmp91, xmask)
    tl.store(out_ptr3 + (x8 + (952560*x7)), tmp105, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkgk6m3alkswp7vxvprec57s4lq2dupsmrt4pywj4arwr36mk3x.py
# Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_1_avgpool, x_174], Original ATen: [aten.avg_pool2d, aten.relu]
# l__mod___cell_0_conv_prev_1x1_path_1_avgpool => avg_pool2d_2
# x_174 => relu_29
triton_poi_fused_avg_pool2d_relu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 42
    x3 = (xindex // 42)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 270
    y1 = (yindex // 270)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (166*x3) + (6889*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (y0 + (270*x5) + (476280*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkbxuphc4dadqtkd2wz5iq377ufyjhkqamwrkk4oe4odmygg2jx.py
# Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_2_avgpool, l__mod___cell_0_conv_prev_1x1_path_2_pad, x_174], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
# l__mod___cell_0_conv_prev_1x1_path_2_avgpool => avg_pool2d_3
# l__mod___cell_0_conv_prev_1x1_path_2_pad => constant_pad_nd_19
# x_174 => relu_29
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 42)
    x2 = xindex % 42
    y4 = yindex
    x5 = xindex
    y0 = yindex % 270
    y1 = (yindex // 270)
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (84 + (2*x2) + (166*x3) + (6889*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (270*x5) + (476280*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3wsbgflykgaijigpb42o6gspxvb6m77ynn47sc2nextuerwb7e.py
# Source Nodes: [cat_32, x_178, x_228, x_left_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_32 => cat_3
# x_178 => relu_31
# x_228 => relu_41
# x_left_1 => add_73, mul_94, mul_95, sub_31
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 216
    x2 = xindex
    y1 = (yindex // 216)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (1764*y0) + (190512*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 216, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-190512) + x2 + (1764*y0) + (190512*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x2 + (1764*y3)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (216*x2) + (381024*y1)), tmp29, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (216*x2) + (381024*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cukftspg4sveba4mfzmrqbkp4p5slmk5qxabbmkkifjw2kqfirnb.py
# Source Nodes: [x_181], Original ATen: [aten.convolution]
# x_181 => convolution_60
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (216*x2) + (381024*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/cole32f6tvwtxavqkmf3uu2bbox6xtoqj6rcfvgzh4mfnrpk6lku.py
# Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_182 => add_77, mul_100, mul_101, sub_33
# x_183 => relu_32
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (216*x2) + (381024*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jy/cjywoffj36u6p7x43hkblon5fzueqdjasifts53ag723gnhwjdam.py
# Source Nodes: [x_comb_iter_0_left_2, x_comb_iter_0_right_2, x_comb_iter_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
# x_comb_iter_0_left_2 => add_79, mul_103, mul_104, sub_34
# x_comb_iter_0_right_2 => max_pool2d_with_indices_6
# x_comb_iter_10 => add_80
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x4 = xindex
    x6 = (xindex // 1764) % 216
    x7 = (xindex // 381024)
    x8 = xindex % 381024
    tmp70 = tl.load(in_ptr1 + (x4), xmask)
    tmp71 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-43) + x4), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-42) + x4), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-41) + x4), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (41 + x4), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (42 + x4), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (43 + x4), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp72 = tmp70 - tmp71
    tmp74 = 0.001
    tmp75 = tmp73 + tmp74
    tmp76 = tl.sqrt(tmp75)
    tmp77 = 1 / tmp76
    tmp78 = 1.0
    tmp79 = tmp77 * tmp78
    tmp80 = tmp72 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp84 + tmp69
    tl.store(out_ptr1 + (x8 + (1905120*x7)), tmp85, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhel3gnpnni4v4sl3qsjtndfr6hlo3wcmocsgcp3ljbjx3rle62.py
# Source Nodes: [x_comb_iter_4_left_1, x_comb_iter_4_right_1, x_comb_iter_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_4_left_1 => add_68, mul_88, mul_89, sub_29
# x_comb_iter_4_right_1 => add_70, mul_91, mul_92, sub_30
# x_comb_iter_9 => add_71
triton_poi_fused__native_batch_norm_legit_no_training_add_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1764) % 108
    x2 = (xindex // 190512)
    x4 = xindex % 190512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x4 + (952560*x2)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyirddt34lpmo3e34pqf44wufssq7lsevnoc67627x2wnxh3u3tz.py
# Source Nodes: [x_175, x_238], Original ATen: [aten.relu]
# x_175 => relu_30
# x_238 => relu_43
triton_poi_fused_relu_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4320
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 540
    y1 = (yindex // 540)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (540*x2) + (952560*y1)), tmp1, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (540*x2) + (952560*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cme6pkbe6fv4zsaycqazy24t5ogtywc774u4qwgxdyzej6pmn3ko.py
# Source Nodes: [x_comb_iter_4_right_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_comb_iter_4_right_2 => add_75, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1764) % 216
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


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlbgeob4cjppbhb3cxomkcsgset35f6ivfjl4cnxr32smfjsw2t.py
# Source Nodes: [x_188, x_198, x_208, x_comb_iter_14, x_comb_iter_1_right_2, x_comb_iter_3_right_2, x_comb_iter_4_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
# x_188 => relu_33
# x_198 => relu_35
# x_208 => relu_37
# x_comb_iter_14 => add_104
# x_comb_iter_1_right_2 => max_pool2d_with_indices_7
# x_comb_iter_3_right_2 => max_pool2d_with_indices_8
# x_comb_iter_4_left_2 => add_103, mul_133, mul_134, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 42)
    x1 = xindex % 42
    x3 = xindex
    y0 = yindex
    y4 = yindex % 216
    y5 = (yindex // 216)
    tmp70 = tl.load(in_ptr0 + (x3 + (1764*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (x3 + (1764*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr5 + (y4), ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-43) + x3 + (1764*y0)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-42) + x3 + (1764*y0)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-41) + x3 + (1764*y0)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x3 + (1764*y0)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x3 + (1764*y0)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x3 + (1764*y0)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (41 + x3 + (1764*y0)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (42 + x3 + (1764*y0)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (43 + x3 + (1764*y0)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp71 = triton_helpers.maximum(0, tmp70)
    tmp74 = tmp72 - tmp73
    tmp76 = 0.001
    tmp77 = tmp75 + tmp76
    tmp78 = tl.sqrt(tmp77)
    tmp79 = 1 / tmp78
    tmp80 = 1.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp74 * tmp81
    tmp84 = tmp82 * tmp83
    tmp86 = tmp84 + tmp85
    tmp87 = tmp86 + tmp70
    tl.store(out_ptr0 + (x3 + (1764*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr1 + (x3 + (1764*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr2 + (y4 + (216*x3) + (381024*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr3 + (y4 + (216*x3) + (381024*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr4 + (y4 + (216*x3) + (381024*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr5 + (x3 + (1764*y4) + (1905120*y5)), tmp87, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvk2hh7knjzhecmglaw2bk5b4wcjet7ew7jx3us37goxly77wk4z.py
# Source Nodes: [x_244, x_294, x_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_244 => relu_45
# x_294 => relu_55
# x_left_2 => add_106, mul_136, mul_137, sub_45
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1764*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (216*x2) + (381024*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (216*x2) + (381024*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlev47ayhlukzz42zahyvw6vq5vyzhu7ava4fo6cnigjghxvlji.py
# Source Nodes: [x_218, x_comb_iter_12, x_comb_iter_2_left_2, x_comb_iter_2_right_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_218 => relu_39
# x_comb_iter_12 => add_94
# x_comb_iter_2_left_2 => add_89, mul_115, mul_116, sub_38
# x_comb_iter_2_right_2 => add_93, mul_121, mul_122, sub_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2 + (1764*y0) + (1905120*y1)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (216*x2) + (381024*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/cso7yfuuktlfrzhf7vjd3djdz2usy2sls47rnbysybxvq63yolc4.py
# Source Nodes: [x_comb_iter_11, x_comb_iter_1_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_11 => add_85
# x_comb_iter_1_left_2 => add_84, mul_109, mul_110, sub_36
triton_poi_fused__native_batch_norm_legit_no_training_add_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1764) % 216
    x2 = (xindex // 381024)
    x4 = xindex % 381024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x4 + (1905120*x2)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl72nx7ypr54nmmjytgubfq3wusnobugms5hwcqltdtlzhmaeuwq.py
# Source Nodes: [x_241, x_304], Original ATen: [aten.relu]
# x_241 => relu_44
# x_304 => relu_57
triton_poi_fused_relu_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (1080*x2) + (1905120*y1)), tmp1, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (1080*x2) + (1905120*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbn4kvi6m54texrnvubrx5m5ykpfmnnh6t273meslt2wuy5skay7.py
# Source Nodes: [x_left_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_left_5 => add_205, mul_262, mul_263, sub_87
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1764) % 432
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


# kernel path: /tmp/torchinductor_youkaichao/lm/clmvc7pxve2aekyiczn6n2o55vzv55ezriyhmqiyxgdxnqntt4dc.py
# Source Nodes: [x_442, x_444], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_442 => relu_87
# x_444 => constant_pad_nd_20
triton_poi_fused_constant_pad_nd_relu_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 2025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 45)
    x2 = xindex % 45
    y4 = yindex
    x5 = xindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-43) + x2 + (42*x3) + (1764*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (432*x5) + (874800*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgire4ycivbwmameebkrd7umazuwaygyqr4pkpgcshvlxmp24qey.py
# Source Nodes: [x_447], Original ATen: [aten.convolution]
# x_447 => convolution_164
triton_poi_fused_convolution_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
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
    tmp0 = tl.load(in_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (432*x2) + (190512*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctweeke3tnys25egniaoz2zgv2hag66inuxmzndtz6z3p7r7gck7.py
# Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_448 => add_209, mul_268, mul_269, sub_89
# x_449 => relu_88
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
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
    tmp0 = tl.load(in_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (432*x2) + (190512*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2pmjgvsehujdz5w33ka5bpewk42dlscy4k66qzyoxyc5ntxgu5.py
# Source Nodes: [x_455, x_comb_iter_0_left_6, x_comb_iter_0_right_6, x_comb_iter_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_455 => constant_pad_nd_21
# x_comb_iter_0_left_6 => add_211, mul_271, mul_272, sub_90
# x_comb_iter_0_right_6 => max_pool2d_with_indices_18
# x_comb_iter_30 => add_212
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21) % 21
    x0 = xindex % 21
    x3 = (xindex // 21)
    x4 = xindex
    x6 = (xindex // 441) % 432
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp57 = tl.load(in_ptr1 + (x4), xmask)
    tmp58 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = 2*x1
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((2*x0) + (84*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, float("-inf"), tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1 + (2*x0)
    tmp10 = tmp9 < tmp1
    tmp11 = tmp2 & tmp10
    tmp12 = tl.load(in_ptr0 + (1 + (2*x0) + (84*x3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, float("-inf"), tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = 2 + (2*x0)
    tmp17 = tmp16 < tmp1
    tmp18 = tmp2 & tmp17
    tmp19 = tl.load(in_ptr0 + (2 + (2*x0) + (84*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp15)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp4
    tmp26 = tl.load(in_ptr0 + (42 + (2*x0) + (84*x3)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, float("-inf"), tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp22)
    tmp30 = tmp24 & tmp10
    tmp31 = tl.load(in_ptr0 + (43 + (2*x0) + (84*x3)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, float("-inf"), tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp33, tmp29)
    tmp35 = tmp24 & tmp17
    tmp36 = tl.load(in_ptr0 + (44 + (2*x0) + (84*x3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full(tmp36.shape, float("-inf"), tmp36.dtype)
    tmp38 = tl.where(tmp35, tmp36, tmp37)
    tmp39 = triton_helpers.maximum(tmp38, tmp34)
    tmp40 = 2 + (2*x1)
    tmp41 = tmp40 < tmp1
    tmp42 = tmp41 & tmp4
    tmp43 = tl.load(in_ptr0 + (84 + (2*x0) + (84*x3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, float("-inf"), tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp41 & tmp10
    tmp48 = tl.load(in_ptr0 + (85 + (2*x0) + (84*x3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, float("-inf"), tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = triton_helpers.maximum(tmp50, tmp46)
    tmp52 = tmp41 & tmp17
    tmp53 = tl.load(in_ptr0 + (86 + (2*x0) + (84*x3)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, float("-inf"), tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = triton_helpers.maximum(tmp55, tmp51)
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp71 + tmp56
    tl.store(out_ptr1 + (x8 + (952560*x7)), tmp72, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4fmlgsbaqouf5gydhbxfvbergfsvhdvvn5sgztijpz6xh75rs3.py
# Source Nodes: [x_439], Original ATen: [aten.relu]
# x_439 => relu_86
triton_poi_fused_relu_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = tl.load(in_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (1080*x2) + (1905120*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhmib6ylvnlunyb4aud7cf5nnvpgjeai2fl2rqyepf2xpzmscu4.py
# Source Nodes: [x_518, x_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_518 => relu_99
# x_right_6 => add_207, mul_265, mul_266, sub_88
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 1764
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1764*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1764*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (432*x2) + (762048*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvd62ni47ddwjlhbihqudomvg2bwlu2c6f32k4zui6fb4ymppiug.py
# Source Nodes: [x_456, x_458], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_456 => relu_89
# x_458 => constant_pad_nd_22
triton_poi_fused_constant_pad_nd_relu_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 2209
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 47)
    x2 = xindex % 47
    y4 = yindex
    x5 = xindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-86) + x2 + (42*x3) + (1764*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (432*x5) + (954288*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca63ae33rqxpjknznnq2mkhspm6fbdpcmb4wahvdxpxhqotdcogx.py
# Source Nodes: [x_482, x_484], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_482 => relu_93
# x_484 => constant_pad_nd_25
triton_poi_fused_constant_pad_nd_relu_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 1849
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 43)
    x2 = xindex % 43
    y4 = yindex
    x5 = xindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (42*x3) + (1764*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + (432*x5) + (798768*y1)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyit3mi6ebej42zkfze3imzw746xgffsea2vnbom6luoi7cnv3f.py
# Source Nodes: [x_494, x_comb_iter_2_left_6, x_comb_iter_2_right_6, x_comb_iter_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_494 => relu_95
# x_comb_iter_2_left_6 => add_221, mul_283, mul_284, sub_94
# x_comb_iter_2_right_6 => add_225, mul_289, mul_290, sub_96
# x_comb_iter_32 => add_226
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
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
    tmp0 = tl.load(in_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2 + (441*y0) + (952560*y1)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (432*x2) + (190512*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6cv3hoym6zbzfhkdxf6zvyfyxc7t77c226okkvv5hxv36xr2hj.py
# Source Nodes: [x_469, x_505, x_comb_iter_1_left_6, x_comb_iter_1_right_6, x_comb_iter_31, x_comb_iter_33, x_comb_iter_3_left_6, x_comb_iter_3_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_469 => constant_pad_nd_23
# x_505 => constant_pad_nd_26
# x_comb_iter_1_left_6 => add_216, mul_277, mul_278, sub_92
# x_comb_iter_1_right_6 => max_pool2d_with_indices_19
# x_comb_iter_31 => add_217
# x_comb_iter_33 => add_231
# x_comb_iter_3_left_6 => add_230, mul_295, mul_296, sub_98
# x_comb_iter_3_right_6 => max_pool2d_with_indices_20
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21) % 21
    x0 = xindex % 21
    x3 = (xindex // 21)
    x4 = xindex
    x6 = (xindex // 441) % 432
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp57 = tl.load(in_ptr1 + (x4), xmask)
    tmp58 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr6 + (x4), xmask)
    tmp74 = tl.load(in_ptr7 + (x6), xmask, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr8 + (x6), xmask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr9 + (x6), xmask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr10 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = 2*x1
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((2*x0) + (84*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, float("-inf"), tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1 + (2*x0)
    tmp10 = tmp9 < tmp1
    tmp11 = tmp2 & tmp10
    tmp12 = tl.load(in_ptr0 + (1 + (2*x0) + (84*x3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, float("-inf"), tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = 2 + (2*x0)
    tmp17 = tmp16 < tmp1
    tmp18 = tmp2 & tmp17
    tmp19 = tl.load(in_ptr0 + (2 + (2*x0) + (84*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp15)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp4
    tmp26 = tl.load(in_ptr0 + (42 + (2*x0) + (84*x3)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, float("-inf"), tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp22)
    tmp30 = tmp24 & tmp10
    tmp31 = tl.load(in_ptr0 + (43 + (2*x0) + (84*x3)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, float("-inf"), tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp33, tmp29)
    tmp35 = tmp24 & tmp17
    tmp36 = tl.load(in_ptr0 + (44 + (2*x0) + (84*x3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full(tmp36.shape, float("-inf"), tmp36.dtype)
    tmp38 = tl.where(tmp35, tmp36, tmp37)
    tmp39 = triton_helpers.maximum(tmp38, tmp34)
    tmp40 = 2 + (2*x1)
    tmp41 = tmp40 < tmp1
    tmp42 = tmp41 & tmp4
    tmp43 = tl.load(in_ptr0 + (84 + (2*x0) + (84*x3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, float("-inf"), tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp41 & tmp10
    tmp48 = tl.load(in_ptr0 + (85 + (2*x0) + (84*x3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, float("-inf"), tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = triton_helpers.maximum(tmp50, tmp46)
    tmp52 = tmp41 & tmp17
    tmp53 = tl.load(in_ptr0 + (86 + (2*x0) + (84*x3)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, float("-inf"), tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = triton_helpers.maximum(tmp55, tmp51)
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp71 + tmp56
    tmp75 = tmp73 - tmp74
    tmp77 = tmp76 + tmp61
    tmp78 = tl.sqrt(tmp77)
    tmp79 = 1 / tmp78
    tmp80 = tmp79 * tmp65
    tmp81 = tmp75 * tmp80
    tmp83 = tmp81 * tmp82
    tmp85 = tmp83 + tmp84
    tmp86 = tmp85 + tmp56
    tl.store(out_ptr2 + (x8 + (952560*x7)), tmp72, xmask)
    tl.store(out_ptr3 + (x8 + (952560*x7)), tmp86, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c627eymjkztoio2x5jgqyau2lpruvd7uyr3j6vnwmwv4jmuhmzvk.py
# Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_1_avgpool, x_523], Original ATen: [aten.avg_pool2d, aten.relu]
# l__mod___cell_5_conv_prev_1x1_path_1_avgpool => avg_pool2d_4
# x_523 => relu_100
triton_poi_fused_avg_pool2d_relu_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 21
    x3 = (xindex // 21)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (84*x3) + (1764*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (y0 + (1080*x5) + (476280*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wk/cwkxirs46m4zorqsqfuvuo6xwaa327yihm6ebxupaad7l7eqvdmg.py
# Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_2_avgpool, l__mod___cell_5_conv_prev_1x1_path_2_pad, x_523], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
# l__mod___cell_5_conv_prev_1x1_path_2_avgpool => avg_pool2d_5
# l__mod___cell_5_conv_prev_1x1_path_2_pad => constant_pad_nd_29
# x_523 => relu_100
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 21)
    x2 = xindex % 21
    y4 = yindex
    x5 = xindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (43 + (2*x2) + (84*x3) + (1764*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (1080*x5) + (476280*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlcsvpdh74pxur6ija3ufcd7sbr3rxthhkya4ugaikux3nfvla2.py
# Source Nodes: [cat_26, x_527, x_577, x_left_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_26 => cat_9
# x_527 => relu_102
# x_577 => relu_112
# x_left_6 => add_240, mul_307, mul_308, sub_102
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 432
    x2 = xindex
    y1 = (yindex // 432)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (441*y0) + (95256*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 432, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-95256) + x2 + (441*y0) + (95256*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x2 + (441*y3)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (432*x2) + (190512*y1)), tmp29, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (432*x2) + (190512*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czo7ofuwimiyprfogts4s6k5fmgkvkarct7q4dtyodfo65zjv4mq.py
# Source Nodes: [x_comb_iter_0_left_7, x_comb_iter_0_right_7, x_comb_iter_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
# x_comb_iter_0_left_7 => add_246, mul_316, mul_317, sub_105
# x_comb_iter_0_right_7 => max_pool2d_with_indices_21
# x_comb_iter_35 => add_247
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21) % 21
    x0 = xindex % 21
    x4 = xindex
    x6 = (xindex // 441) % 432
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp70 = tl.load(in_ptr1 + (x4), xmask)
    tmp71 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-22) + x4), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-21) + x4), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-20) + x4), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (20 + x4), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (21 + x4), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (22 + x4), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp72 = tmp70 - tmp71
    tmp74 = 0.001
    tmp75 = tmp73 + tmp74
    tmp76 = tl.sqrt(tmp75)
    tmp77 = 1 / tmp76
    tmp78 = 1.0
    tmp79 = tmp77 * tmp78
    tmp80 = tmp72 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp84 + tmp69
    tl.store(out_ptr1 + (x8 + (952560*x7)), tmp85, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csnijijjhlh32o7hebdw6n2zoxamxitpwif4yfnkzjnaq25a55kl.py
# Source Nodes: [x_comb_iter_34, x_comb_iter_4_left_6, x_comb_iter_4_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_34 => add_238
# x_comb_iter_4_left_6 => add_235, mul_301, mul_302, sub_100
# x_comb_iter_4_right_6 => add_237, mul_304, mul_305, sub_101
triton_poi_fused__native_batch_norm_legit_no_training_add_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 441) % 432
    x2 = (xindex // 190512)
    x4 = xindex % 190512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x4 + (952560*x2)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctatxtrog27bsfvlyc4nzhss3yfrudgev2giuuqhl5bwojlmhfb3.py
# Source Nodes: [x_524, x_587], Original ATen: [aten.relu]
# x_524 => relu_101
# x_587 => relu_114
triton_poi_fused_relu_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    tmp0 = tl.load(in_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (2160*x2) + (952560*y1)), tmp1, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (2160*x2) + (952560*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3hkrvqf3kix46vygfcrbr7nxyvj4ol4c22v733l4by2zbsjep5.py
# Source Nodes: [x_comb_iter_4_right_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_comb_iter_4_right_7 => add_242, mul_310, mul_311, sub_103
triton_poi_fused__native_batch_norm_legit_no_training_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 441) % 432
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


# kernel path: /tmp/torchinductor_youkaichao/qu/cqud77chndhmjmnpsm72nq7o4th4e3af5fd626enrrjhjoey5tcq.py
# Source Nodes: [x_537, x_547, x_557, x_comb_iter_1_right_7, x_comb_iter_39, x_comb_iter_3_right_7, x_comb_iter_4_left_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
# x_537 => relu_104
# x_547 => relu_106
# x_557 => relu_108
# x_comb_iter_1_right_7 => max_pool2d_with_indices_22
# x_comb_iter_39 => add_271
# x_comb_iter_3_right_7 => max_pool2d_with_indices_23
# x_comb_iter_4_left_7 => add_270, mul_346, mul_347, sub_115
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 21)
    x1 = xindex % 21
    x3 = xindex
    y0 = yindex
    y4 = yindex % 432
    y5 = (yindex // 432)
    tmp70 = tl.load(in_ptr0 + (x3 + (441*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (x3 + (441*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr5 + (y4), ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-22) + x3 + (441*y0)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-21) + x3 + (441*y0)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-20) + x3 + (441*y0)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x3 + (441*y0)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x3 + (441*y0)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x3 + (441*y0)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (20 + x3 + (441*y0)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (21 + x3 + (441*y0)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (22 + x3 + (441*y0)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp71 = triton_helpers.maximum(0, tmp70)
    tmp74 = tmp72 - tmp73
    tmp76 = 0.001
    tmp77 = tmp75 + tmp76
    tmp78 = tl.sqrt(tmp77)
    tmp79 = 1 / tmp78
    tmp80 = 1.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp74 * tmp81
    tmp84 = tmp82 * tmp83
    tmp86 = tmp84 + tmp85
    tmp87 = tmp86 + tmp70
    tl.store(out_ptr0 + (x3 + (441*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr1 + (x3 + (441*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr2 + (y4 + (432*x3) + (190512*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr3 + (y4 + (432*x3) + (190512*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr4 + (y4 + (432*x3) + (190512*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr5 + (x3 + (441*y4) + (952560*y5)), tmp87, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5uab67j5qorfcj4xgcf6eoynnmmw7saqf2x47atflrvpt5j6nca.py
# Source Nodes: [x_593, x_643, x_left_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_593 => relu_116
# x_643 => relu_126
# x_left_7 => add_273, mul_349, mul_350, sub_116
triton_poi_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (441*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (432*x2) + (190512*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (432*x2) + (190512*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2igvrnoxzfjdh4ufsjtgqczdpqjock7wwby3rlt2ad3dn5dmrx.py
# Source Nodes: [x_comb_iter_1_left_7, x_comb_iter_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_1_left_7 => add_251, mul_322, mul_323, sub_107
# x_comb_iter_36 => add_252
triton_poi_fused__native_batch_norm_legit_no_training_add_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 441) % 432
    x2 = (xindex // 190512)
    x4 = xindex % 190512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x4 + (952560*x2)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz24gxocsltakvhryuvqm4asi3jmqqbcwwohcpj5nbllblfmjgj7.py
# Source Nodes: [x_left_9], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_left_9 => add_339, mul_433, mul_434, sub_144
triton_poi_fused__native_batch_norm_legit_no_training_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 441) % 864
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjs4fuhapnz5lyedh64s57a5muwxec7lcbykux54gk5d2wezslfa.py
# Source Nodes: [x_725, x_727], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_725 => relu_144
# x_727 => constant_pad_nd_30
triton_poi_fused_constant_pad_nd_relu_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 25)
    x2 = xindex % 25
    y4 = yindex
    x5 = xindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-44) + x2 + (21*x3) + (441*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (864*x5) + (540000*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ub/cubkopfrbyapqq2ujixzk7kjwiiy4rumw6hqtehznlseyffpt25g.py
# Source Nodes: [x_730], Original ATen: [aten.convolution]
# x_730 => convolution_270
triton_poi_fused_convolution_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (864*x2) + (104544*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kmcdw25ifttscqqfv6emtglwrrhcgybe2qpla5oevotiyvdo5l.py
# Source Nodes: [x_731, x_732], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_731 => add_343, mul_439, mul_440, sub_146
# x_732 => relu_145
triton_poi_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (864*x2) + (104544*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhxmg52mkys2a2z6wlm4ppwskxcocmppiqix2nflhlifz5hduwi.py
# Source Nodes: [x_738, x_comb_iter_0_left_10, x_comb_iter_0_right_10, x_comb_iter_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_738 => constant_pad_nd_31
# x_comb_iter_0_left_10 => add_345, mul_442, mul_443, sub_147
# x_comb_iter_0_right_10 => max_pool2d_with_indices_30
# x_comb_iter_50 => add_346
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 11) % 11
    x0 = xindex % 11
    x2 = (xindex // 121)
    x4 = xindex
    x6 = (xindex // 121) % 864
    x7 = (xindex // 104544)
    x8 = xindex % 104544
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-22) + (2*x0) + (42*x1) + (441*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-21) + (2*x0) + (42*x1) + (441*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-20) + (2*x0) + (42*x1) + (441*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (42*x1) + (441*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (42*x1) + (441*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (42*x1) + (441*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (20 + (2*x0) + (42*x1) + (441*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (21 + (2*x0) + (42*x1) + (441*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (22 + (2*x0) + (42*x1) + (441*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tl.store(out_ptr1 + (x8 + (522720*x7)), tmp91, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphnoqlxyat7e4h3kvawm7asmbx2xljnhhur2fypvsqhtm2hfwz2.py
# Source Nodes: [x_722], Original ATen: [aten.relu]
# x_722 => relu_143
triton_poi_fused_relu_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    tmp0 = tl.load(in_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (2160*x2) + (952560*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camdmi4jzsbz5ktoypwwrbd4uxvfzftgylqs4iyvnikv4gqmmiz2.py
# Source Nodes: [x_801, x_right_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_801 => relu_156
# x_right_10 => add_341, mul_436, mul_437, sub_145
triton_poi_fused__native_batch_norm_legit_no_training_relu_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (441*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (441*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (864*x2) + (381024*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cif72exbnly2mhokwk2jloo4cw6w7maknpxorgqzx4jef2ym3hgf.py
# Source Nodes: [x_739, x_741], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_739 => relu_146
# x_741 => constant_pad_nd_32
triton_poi_fused_constant_pad_nd_relu_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 729
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 27)
    x2 = xindex % 27
    y4 = yindex
    x5 = xindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = (-3) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-66) + x2 + (21*x3) + (441*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (864*x5) + (629856*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokcyajmxoyn25mseylhwzvpfdvhdluig7ditcipetlnqlag23cr.py
# Source Nodes: [x_765, x_767], Original ATen: [aten.constant_pad_nd, aten.relu]
# x_765 => relu_150
# x_767 => constant_pad_nd_35
triton_poi_fused_constant_pad_nd_relu_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 529
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 23)
    x2 = xindex % 23
    y4 = yindex
    x5 = xindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-22) + x2 + (21*x3) + (441*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tl.store(out_ptr0 + (y0 + (864*x5) + (457056*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tp/ctp76rzor4s4f7zaxs7wd3ewrktekzw6lr3wlyfdfulujixi7xxw.py
# Source Nodes: [x_777, x_comb_iter_2_left_10, x_comb_iter_2_right_10, x_comb_iter_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_777 => relu_152
# x_comb_iter_2_left_10 => add_355, mul_454, mul_455, sub_151
# x_comb_iter_2_right_10 => add_359, mul_460, mul_461, sub_153
# x_comb_iter_52 => add_360
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2 + (121*y0) + (522720*y1)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (864*x2) + (104544*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmko557qimhgptfu2yhidbpseluueauosz5vhxe5yp6amh3r524f.py
# Source Nodes: [x_752, x_788, x_comb_iter_1_left_10, x_comb_iter_1_right_10, x_comb_iter_3_left_10, x_comb_iter_3_right_10, x_comb_iter_51, x_comb_iter_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_752 => constant_pad_nd_33
# x_788 => constant_pad_nd_36
# x_comb_iter_1_left_10 => add_350, mul_448, mul_449, sub_149
# x_comb_iter_1_right_10 => max_pool2d_with_indices_31
# x_comb_iter_3_left_10 => add_364, mul_466, mul_467, sub_155
# x_comb_iter_3_right_10 => max_pool2d_with_indices_32
# x_comb_iter_51 => add_351
# x_comb_iter_53 => add_365
triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 11) % 11
    x0 = xindex % 11
    x2 = (xindex // 121)
    x4 = xindex
    x6 = (xindex // 121) % 864
    x7 = (xindex // 104544)
    x8 = xindex % 104544
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x4), xmask)
    tmp93 = tl.load(in_ptr7 + (x6), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr8 + (x6), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr9 + (x6), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr10 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-22) + (2*x0) + (42*x1) + (441*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-21) + (2*x0) + (42*x1) + (441*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-20) + (2*x0) + (42*x1) + (441*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (42*x1) + (441*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (42*x1) + (441*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (42*x1) + (441*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (20 + (2*x0) + (42*x1) + (441*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (21 + (2*x0) + (42*x1) + (441*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (22 + (2*x0) + (42*x1) + (441*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tmp94 = tmp92 - tmp93
    tmp96 = tmp95 + tmp80
    tmp97 = tl.sqrt(tmp96)
    tmp98 = 1 / tmp97
    tmp99 = tmp98 * tmp84
    tmp100 = tmp94 * tmp99
    tmp102 = tmp100 * tmp101
    tmp104 = tmp102 + tmp103
    tmp105 = tmp104 + tmp75
    tl.store(out_ptr2 + (x8 + (522720*x7)), tmp91, xmask)
    tl.store(out_ptr3 + (x8 + (522720*x7)), tmp105, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtpir4xrqj2zwadtq3fkdodhwvf436mr6sbgxafogfbwnhrv2e7.py
# Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_1_avgpool, x_806], Original ATen: [aten.avg_pool2d, aten.relu]
# l__mod___cell_9_conv_prev_1x1_path_1_avgpool => avg_pool2d_6
# x_806 => relu_157
triton_poi_fused_avg_pool2d_relu_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 11
    x3 = (xindex // 11)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (42*x3) + (441*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (y0 + (2160*x5) + (261360*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifl7vxttkhgclyu3xqvduqzukj64f7eh3harj7ekiiwreavkzst.py
# Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_2_avgpool, l__mod___cell_9_conv_prev_1x1_path_2_pad, x_806], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
# l__mod___cell_9_conv_prev_1x1_path_2_avgpool => avg_pool2d_7
# l__mod___cell_9_conv_prev_1x1_path_2_pad => constant_pad_nd_39
# x_806 => relu_157
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 11)
    x2 = xindex % 11
    y4 = yindex
    x5 = xindex
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (22 + (2*x2) + (42*x3) + (441*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (2160*x5) + (261360*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eg2edtaq4kffeed4nt72olvipgepgbq3cy7ljawkty2w3bpjmp.py
# Source Nodes: [cat_21, x_810, x_860, x_left_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_21 => cat_14
# x_810 => relu_159
# x_860 => relu_169
# x_left_10 => add_374, mul_478, mul_479, sub_159
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 864
    x2 = xindex
    y1 = (yindex // 864)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (121*y0) + (52272*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 864, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-52272) + x2 + (121*y0) + (52272*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x2 + (121*y3)), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (864*x2) + (104544*y1)), tmp29, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (864*x2) + (104544*y1)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuw2c5yoxinrj5jqe3bj5vx3bvgb7xuvncfck4c2arwebdesm5u2.py
# Source Nodes: [x_comb_iter_0_left_11, x_comb_iter_0_right_11, x_comb_iter_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
# x_comb_iter_0_left_11 => add_380, mul_487, mul_488, sub_162
# x_comb_iter_0_right_11 => max_pool2d_with_indices_33
# x_comb_iter_55 => add_381
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 11) % 11
    x0 = xindex % 11
    x4 = xindex
    x6 = (xindex // 121) % 864
    x7 = (xindex // 104544)
    x8 = xindex % 104544
    tmp70 = tl.load(in_ptr1 + (x4), xmask)
    tmp71 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 11, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-12) + x4), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11) + x4), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-10) + x4), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (10 + x4), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (11 + x4), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (12 + x4), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp72 = tmp70 - tmp71
    tmp74 = 0.001
    tmp75 = tmp73 + tmp74
    tmp76 = tl.sqrt(tmp75)
    tmp77 = 1 / tmp76
    tmp78 = 1.0
    tmp79 = tmp77 * tmp78
    tmp80 = tmp72 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp84 + tmp69
    tl.store(out_ptr1 + (x8 + (522720*x7)), tmp85, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc2utye5g2iw3uer5f2awlhtmlrtcp66c66gzafbmez7ylcwqg6c.py
# Source Nodes: [x_comb_iter_4_left_10, x_comb_iter_4_right_10, x_comb_iter_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_4_left_10 => add_369, mul_472, mul_473, sub_157
# x_comb_iter_4_right_10 => add_371, mul_475, mul_476, sub_158
# x_comb_iter_54 => add_372
triton_poi_fused__native_batch_norm_legit_no_training_add_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 121) % 864
    x2 = (xindex // 104544)
    x4 = xindex % 104544
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x4 + (522720*x2)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cin7cakdeto4pl5q4z6mrfhqe4aep22ehjhvzkxanpmlmolcx76r.py
# Source Nodes: [x_807, x_870], Original ATen: [aten.relu]
# x_807 => relu_158
# x_870 => relu_171
triton_poi_fused_relu_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 34560
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4320
    y1 = (yindex // 4320)
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (4320*x2) + (522720*y1)), tmp1, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (4320*x2) + (522720*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43d6bh7bkjoyi73yhotznt2uwqkom5r2kq7gx2sgpjy6qzdenwt.py
# Source Nodes: [x_comb_iter_4_right_11], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_comb_iter_4_right_11 => add_376, mul_481, mul_482, sub_160
triton_poi_fused__native_batch_norm_legit_no_training_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_80', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 121) % 864
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


# kernel path: /tmp/torchinductor_youkaichao/54/c54huqwnru4wojlvc22pwevaa7nthvvzkssfu2w3iddhnvh2ffq5.py
# Source Nodes: [x_820, x_830, x_840, x_comb_iter_1_right_11, x_comb_iter_3_right_11, x_comb_iter_4_left_11, x_comb_iter_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
# x_820 => relu_161
# x_830 => relu_163
# x_840 => relu_165
# x_comb_iter_1_right_11 => max_pool2d_with_indices_34
# x_comb_iter_3_right_11 => max_pool2d_with_indices_35
# x_comb_iter_4_left_11 => add_404, mul_517, mul_518, sub_172
# x_comb_iter_59 => add_405
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 11)
    x1 = xindex % 11
    x3 = xindex
    y0 = yindex
    y4 = yindex % 864
    y5 = (yindex // 864)
    tmp70 = tl.load(in_ptr0 + (x3 + (121*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (x3 + (121*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr5 + (y4), ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 11, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-12) + x3 + (121*y0)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11) + x3 + (121*y0)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-10) + x3 + (121*y0)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x3 + (121*y0)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x3 + (121*y0)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x3 + (121*y0)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (10 + x3 + (121*y0)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (11 + x3 + (121*y0)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (12 + x3 + (121*y0)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp71 = triton_helpers.maximum(0, tmp70)
    tmp74 = tmp72 - tmp73
    tmp76 = 0.001
    tmp77 = tmp75 + tmp76
    tmp78 = tl.sqrt(tmp77)
    tmp79 = 1 / tmp78
    tmp80 = 1.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp74 * tmp81
    tmp84 = tmp82 * tmp83
    tmp86 = tmp84 + tmp85
    tmp87 = tmp86 + tmp70
    tl.store(out_ptr0 + (x3 + (121*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr1 + (x3 + (121*y0)), tmp69, xmask & ymask)
    tl.store(out_ptr2 + (y4 + (864*x3) + (104544*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr3 + (y4 + (864*x3) + (104544*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr4 + (y4 + (864*x3) + (104544*y5)), tmp71, xmask & ymask)
    tl.store(out_ptr5 + (x3 + (121*y4) + (522720*y5)), tmp87, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogqj2lsut3dfjnkcciujljbqw2pv74xa34yuh6xmdcqzaqyru7b.py
# Source Nodes: [x_876, x_926, x_left_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_876 => relu_173
# x_926 => relu_183
# x_left_11 => add_407, mul_520, mul_521, sub_173
triton_poi_fused__native_batch_norm_legit_no_training_relu_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 864
    y1 = (yindex // 864)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (121*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (864*x2) + (104544*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (864*x2) + (104544*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenhjbzbyg3panyrnekv2ne4rdafd7t64zgap5vf3duvrzpyzpi2.py
# Source Nodes: [x_comb_iter_1_left_11, x_comb_iter_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_comb_iter_1_left_11 => add_385, mul_493, mul_494, sub_164
# x_comb_iter_56 => add_386
triton_poi_fused__native_batch_norm_legit_no_training_add_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 121) % 864
    x2 = (xindex // 104544)
    x4 = xindex % 104544
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x4 + (522720*x2)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpka4veopueoxyllipbolkix3aslubrv2nb3sc5tgp56exiqyzu.py
# Source Nodes: [x_939], Original ATen: [aten.relu]
# x_939 => relu_186
triton_poi_fused_relu_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 34560
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4320
    y1 = (yindex // 4320)
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (y0 + (4320*x2) + (522720*y1)), tmp1, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbvodzi2ksbfmjvyomdz433mjjobqyrbpojpwgkm5pryurhkjuh.py
# Source Nodes: [x_1003, x_1004], Original ATen: [aten.mean, aten.relu]
# x_1003 => relu_199
# x_1004 => mean
triton_per_fused_mean_relu_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_relu_85', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 34560
    rnumel = 121
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (121*x0)), rmask & xmask, other=0.0)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 121.0
    tmp7 = tmp5 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1 = args
    args.clear()
    assert_size_stride(arg0_1, (96, ), (1, ))
    assert_size_stride(arg1_1, (96, ), (1, ))
    assert_size_stride(arg2_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg3_1, (54, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg4_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg5_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg6_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg8_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg9_1, (108, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg10_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg11_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg12_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg14_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg15_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg16_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg17_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg18_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg19_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg20_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg21_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg22_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg23_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg24_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg25_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg26_1, (96, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg27_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg28_1, (54, ), (1, ))
    assert_size_stride(arg29_1, (54, ), (1, ))
    assert_size_stride(arg30_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg31_1, (54, ), (1, ))
    assert_size_stride(arg32_1, (54, ), (1, ))
    assert_size_stride(arg33_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg34_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg35_1, (54, ), (1, ))
    assert_size_stride(arg36_1, (54, ), (1, ))
    assert_size_stride(arg37_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg38_1, (54, ), (1, ))
    assert_size_stride(arg39_1, (54, ), (1, ))
    assert_size_stride(arg40_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg41_1, (54, ), (1, ))
    assert_size_stride(arg42_1, (54, ), (1, ))
    assert_size_stride(arg43_1, (54, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg44_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg45_1, (54, ), (1, ))
    assert_size_stride(arg46_1, (54, ), (1, ))
    assert_size_stride(arg47_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg48_1, (54, ), (1, ))
    assert_size_stride(arg49_1, (54, ), (1, ))
    assert_size_stride(arg50_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg51_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg52_1, (54, ), (1, ))
    assert_size_stride(arg53_1, (54, ), (1, ))
    assert_size_stride(arg54_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg55_1, (54, ), (1, ))
    assert_size_stride(arg56_1, (54, ), (1, ))
    assert_size_stride(arg57_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg58_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg59_1, (54, ), (1, ))
    assert_size_stride(arg60_1, (54, ), (1, ))
    assert_size_stride(arg61_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg62_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg63_1, (54, ), (1, ))
    assert_size_stride(arg64_1, (54, ), (1, ))
    assert_size_stride(arg65_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg66_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg67_1, (54, ), (1, ))
    assert_size_stride(arg68_1, (54, ), (1, ))
    assert_size_stride(arg69_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg70_1, (54, ), (1, ))
    assert_size_stride(arg71_1, (54, ), (1, ))
    assert_size_stride(arg72_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg74_1, (54, ), (1, ))
    assert_size_stride(arg75_1, (54, ), (1, ))
    assert_size_stride(arg76_1, (54, ), (1, ))
    assert_size_stride(arg77_1, (54, ), (1, ))
    assert_size_stride(arg78_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg79_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg80_1, (108, ), (1, ))
    assert_size_stride(arg81_1, (108, ), (1, ))
    assert_size_stride(arg82_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg83_1, (108, ), (1, ))
    assert_size_stride(arg84_1, (108, ), (1, ))
    assert_size_stride(arg85_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg86_1, (108, ), (1, ))
    assert_size_stride(arg87_1, (108, ), (1, ))
    assert_size_stride(arg88_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg89_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg90_1, (108, ), (1, ))
    assert_size_stride(arg91_1, (108, ), (1, ))
    assert_size_stride(arg92_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg93_1, (108, ), (1, ))
    assert_size_stride(arg94_1, (108, ), (1, ))
    assert_size_stride(arg95_1, (108, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg96_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg97_1, (108, ), (1, ))
    assert_size_stride(arg98_1, (108, ), (1, ))
    assert_size_stride(arg99_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg100_1, (108, ), (1, ))
    assert_size_stride(arg101_1, (108, ), (1, ))
    assert_size_stride(arg102_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg103_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg104_1, (108, ), (1, ))
    assert_size_stride(arg105_1, (108, ), (1, ))
    assert_size_stride(arg106_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg107_1, (108, ), (1, ))
    assert_size_stride(arg108_1, (108, ), (1, ))
    assert_size_stride(arg109_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg110_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg111_1, (108, ), (1, ))
    assert_size_stride(arg112_1, (108, ), (1, ))
    assert_size_stride(arg113_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg114_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg115_1, (108, ), (1, ))
    assert_size_stride(arg116_1, (108, ), (1, ))
    assert_size_stride(arg117_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg119_1, (108, ), (1, ))
    assert_size_stride(arg120_1, (108, ), (1, ))
    assert_size_stride(arg121_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg122_1, (108, ), (1, ))
    assert_size_stride(arg123_1, (108, ), (1, ))
    assert_size_stride(arg124_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg125_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg126_1, (108, ), (1, ))
    assert_size_stride(arg127_1, (108, ), (1, ))
    assert_size_stride(arg128_1, (108, ), (1, ))
    assert_size_stride(arg129_1, (108, ), (1, ))
    assert_size_stride(arg130_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg131_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg132_1, (216, ), (1, ))
    assert_size_stride(arg133_1, (216, ), (1, ))
    assert_size_stride(arg134_1, (216, 540, 1, 1), (540, 1, 1, 1))
    assert_size_stride(arg135_1, (216, ), (1, ))
    assert_size_stride(arg136_1, (216, ), (1, ))
    assert_size_stride(arg137_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg138_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg139_1, (216, ), (1, ))
    assert_size_stride(arg140_1, (216, ), (1, ))
    assert_size_stride(arg141_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg142_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg143_1, (216, ), (1, ))
    assert_size_stride(arg144_1, (216, ), (1, ))
    assert_size_stride(arg145_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg146_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg147_1, (216, ), (1, ))
    assert_size_stride(arg148_1, (216, ), (1, ))
    assert_size_stride(arg149_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg150_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg151_1, (216, ), (1, ))
    assert_size_stride(arg152_1, (216, ), (1, ))
    assert_size_stride(arg153_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg154_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg155_1, (216, ), (1, ))
    assert_size_stride(arg156_1, (216, ), (1, ))
    assert_size_stride(arg157_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg158_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg159_1, (216, ), (1, ))
    assert_size_stride(arg160_1, (216, ), (1, ))
    assert_size_stride(arg161_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg162_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg163_1, (216, ), (1, ))
    assert_size_stride(arg164_1, (216, ), (1, ))
    assert_size_stride(arg165_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg166_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg167_1, (216, ), (1, ))
    assert_size_stride(arg168_1, (216, ), (1, ))
    assert_size_stride(arg169_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg170_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg171_1, (216, ), (1, ))
    assert_size_stride(arg172_1, (216, ), (1, ))
    assert_size_stride(arg173_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg174_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg175_1, (216, ), (1, ))
    assert_size_stride(arg176_1, (216, ), (1, ))
    assert_size_stride(arg177_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg178_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg179_1, (216, ), (1, ))
    assert_size_stride(arg180_1, (216, ), (1, ))
    assert_size_stride(arg181_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg182_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg183_1, (216, ), (1, ))
    assert_size_stride(arg184_1, (216, ), (1, ))
    assert_size_stride(arg185_1, (216, 540, 1, 1), (540, 1, 1, 1))
    assert_size_stride(arg186_1, (216, ), (1, ))
    assert_size_stride(arg187_1, (216, ), (1, ))
    assert_size_stride(arg188_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg189_1, (216, ), (1, ))
    assert_size_stride(arg190_1, (216, ), (1, ))
    assert_size_stride(arg191_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg192_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg193_1, (216, ), (1, ))
    assert_size_stride(arg194_1, (216, ), (1, ))
    assert_size_stride(arg195_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg196_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg197_1, (216, ), (1, ))
    assert_size_stride(arg198_1, (216, ), (1, ))
    assert_size_stride(arg199_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg200_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg201_1, (216, ), (1, ))
    assert_size_stride(arg202_1, (216, ), (1, ))
    assert_size_stride(arg203_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg204_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg205_1, (216, ), (1, ))
    assert_size_stride(arg206_1, (216, ), (1, ))
    assert_size_stride(arg207_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg208_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg209_1, (216, ), (1, ))
    assert_size_stride(arg210_1, (216, ), (1, ))
    assert_size_stride(arg211_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg212_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg213_1, (216, ), (1, ))
    assert_size_stride(arg214_1, (216, ), (1, ))
    assert_size_stride(arg215_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg216_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg217_1, (216, ), (1, ))
    assert_size_stride(arg218_1, (216, ), (1, ))
    assert_size_stride(arg219_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg220_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg221_1, (216, ), (1, ))
    assert_size_stride(arg222_1, (216, ), (1, ))
    assert_size_stride(arg223_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg224_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg225_1, (216, ), (1, ))
    assert_size_stride(arg226_1, (216, ), (1, ))
    assert_size_stride(arg227_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg228_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg229_1, (216, ), (1, ))
    assert_size_stride(arg230_1, (216, ), (1, ))
    assert_size_stride(arg231_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg232_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg233_1, (216, ), (1, ))
    assert_size_stride(arg234_1, (216, ), (1, ))
    assert_size_stride(arg235_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg236_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg237_1, (216, ), (1, ))
    assert_size_stride(arg238_1, (216, ), (1, ))
    assert_size_stride(arg239_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg240_1, (216, ), (1, ))
    assert_size_stride(arg241_1, (216, ), (1, ))
    assert_size_stride(arg242_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg243_1, (216, ), (1, ))
    assert_size_stride(arg244_1, (216, ), (1, ))
    assert_size_stride(arg245_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg246_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg247_1, (216, ), (1, ))
    assert_size_stride(arg248_1, (216, ), (1, ))
    assert_size_stride(arg249_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg250_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg251_1, (216, ), (1, ))
    assert_size_stride(arg252_1, (216, ), (1, ))
    assert_size_stride(arg253_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg254_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg255_1, (216, ), (1, ))
    assert_size_stride(arg256_1, (216, ), (1, ))
    assert_size_stride(arg257_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg258_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg259_1, (216, ), (1, ))
    assert_size_stride(arg260_1, (216, ), (1, ))
    assert_size_stride(arg261_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg262_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg263_1, (216, ), (1, ))
    assert_size_stride(arg264_1, (216, ), (1, ))
    assert_size_stride(arg265_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg266_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg267_1, (216, ), (1, ))
    assert_size_stride(arg268_1, (216, ), (1, ))
    assert_size_stride(arg269_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg270_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg271_1, (216, ), (1, ))
    assert_size_stride(arg272_1, (216, ), (1, ))
    assert_size_stride(arg273_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg274_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg275_1, (216, ), (1, ))
    assert_size_stride(arg276_1, (216, ), (1, ))
    assert_size_stride(arg277_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg278_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg279_1, (216, ), (1, ))
    assert_size_stride(arg280_1, (216, ), (1, ))
    assert_size_stride(arg281_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg282_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg283_1, (216, ), (1, ))
    assert_size_stride(arg284_1, (216, ), (1, ))
    assert_size_stride(arg285_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg286_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg287_1, (216, ), (1, ))
    assert_size_stride(arg288_1, (216, ), (1, ))
    assert_size_stride(arg289_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg290_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg291_1, (216, ), (1, ))
    assert_size_stride(arg292_1, (216, ), (1, ))
    assert_size_stride(arg293_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg294_1, (216, ), (1, ))
    assert_size_stride(arg295_1, (216, ), (1, ))
    assert_size_stride(arg296_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg297_1, (216, ), (1, ))
    assert_size_stride(arg298_1, (216, ), (1, ))
    assert_size_stride(arg299_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg300_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg301_1, (216, ), (1, ))
    assert_size_stride(arg302_1, (216, ), (1, ))
    assert_size_stride(arg303_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg304_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg305_1, (216, ), (1, ))
    assert_size_stride(arg306_1, (216, ), (1, ))
    assert_size_stride(arg307_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg308_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg309_1, (216, ), (1, ))
    assert_size_stride(arg310_1, (216, ), (1, ))
    assert_size_stride(arg311_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg312_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg313_1, (216, ), (1, ))
    assert_size_stride(arg314_1, (216, ), (1, ))
    assert_size_stride(arg315_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg316_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg317_1, (216, ), (1, ))
    assert_size_stride(arg318_1, (216, ), (1, ))
    assert_size_stride(arg319_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg320_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg321_1, (216, ), (1, ))
    assert_size_stride(arg322_1, (216, ), (1, ))
    assert_size_stride(arg323_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg324_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg325_1, (216, ), (1, ))
    assert_size_stride(arg326_1, (216, ), (1, ))
    assert_size_stride(arg327_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg328_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg329_1, (216, ), (1, ))
    assert_size_stride(arg330_1, (216, ), (1, ))
    assert_size_stride(arg331_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg332_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg333_1, (216, ), (1, ))
    assert_size_stride(arg334_1, (216, ), (1, ))
    assert_size_stride(arg335_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg336_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg337_1, (216, ), (1, ))
    assert_size_stride(arg338_1, (216, ), (1, ))
    assert_size_stride(arg339_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg340_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg341_1, (216, ), (1, ))
    assert_size_stride(arg342_1, (216, ), (1, ))
    assert_size_stride(arg343_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg344_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg345_1, (216, ), (1, ))
    assert_size_stride(arg346_1, (216, ), (1, ))
    assert_size_stride(arg347_1, (432, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg348_1, (432, ), (1, ))
    assert_size_stride(arg349_1, (432, ), (1, ))
    assert_size_stride(arg350_1, (432, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg351_1, (432, ), (1, ))
    assert_size_stride(arg352_1, (432, ), (1, ))
    assert_size_stride(arg353_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg354_1, (432, ), (1, ))
    assert_size_stride(arg355_1, (432, ), (1, ))
    assert_size_stride(arg356_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg357_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg358_1, (432, ), (1, ))
    assert_size_stride(arg359_1, (432, ), (1, ))
    assert_size_stride(arg360_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg361_1, (432, ), (1, ))
    assert_size_stride(arg362_1, (432, ), (1, ))
    assert_size_stride(arg363_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg364_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg365_1, (432, ), (1, ))
    assert_size_stride(arg366_1, (432, ), (1, ))
    assert_size_stride(arg367_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg368_1, (432, ), (1, ))
    assert_size_stride(arg369_1, (432, ), (1, ))
    assert_size_stride(arg370_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg371_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg372_1, (432, ), (1, ))
    assert_size_stride(arg373_1, (432, ), (1, ))
    assert_size_stride(arg374_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg375_1, (432, ), (1, ))
    assert_size_stride(arg376_1, (432, ), (1, ))
    assert_size_stride(arg377_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg378_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg379_1, (432, ), (1, ))
    assert_size_stride(arg380_1, (432, ), (1, ))
    assert_size_stride(arg381_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg382_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg383_1, (432, ), (1, ))
    assert_size_stride(arg384_1, (432, ), (1, ))
    assert_size_stride(arg385_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg386_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg387_1, (432, ), (1, ))
    assert_size_stride(arg388_1, (432, ), (1, ))
    assert_size_stride(arg389_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg390_1, (432, ), (1, ))
    assert_size_stride(arg391_1, (432, ), (1, ))
    assert_size_stride(arg392_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg393_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg394_1, (432, ), (1, ))
    assert_size_stride(arg395_1, (432, ), (1, ))
    assert_size_stride(arg396_1, (432, ), (1, ))
    assert_size_stride(arg397_1, (432, ), (1, ))
    assert_size_stride(arg398_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg399_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg400_1, (432, ), (1, ))
    assert_size_stride(arg401_1, (432, ), (1, ))
    assert_size_stride(arg402_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg403_1, (432, ), (1, ))
    assert_size_stride(arg404_1, (432, ), (1, ))
    assert_size_stride(arg405_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg406_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg407_1, (432, ), (1, ))
    assert_size_stride(arg408_1, (432, ), (1, ))
    assert_size_stride(arg409_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg410_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg411_1, (432, ), (1, ))
    assert_size_stride(arg412_1, (432, ), (1, ))
    assert_size_stride(arg413_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg414_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg415_1, (432, ), (1, ))
    assert_size_stride(arg416_1, (432, ), (1, ))
    assert_size_stride(arg417_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg418_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg419_1, (432, ), (1, ))
    assert_size_stride(arg420_1, (432, ), (1, ))
    assert_size_stride(arg421_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg422_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg423_1, (432, ), (1, ))
    assert_size_stride(arg424_1, (432, ), (1, ))
    assert_size_stride(arg425_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg426_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg427_1, (432, ), (1, ))
    assert_size_stride(arg428_1, (432, ), (1, ))
    assert_size_stride(arg429_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg430_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg431_1, (432, ), (1, ))
    assert_size_stride(arg432_1, (432, ), (1, ))
    assert_size_stride(arg433_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg434_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg435_1, (432, ), (1, ))
    assert_size_stride(arg436_1, (432, ), (1, ))
    assert_size_stride(arg437_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg438_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg439_1, (432, ), (1, ))
    assert_size_stride(arg440_1, (432, ), (1, ))
    assert_size_stride(arg441_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg442_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg443_1, (432, ), (1, ))
    assert_size_stride(arg444_1, (432, ), (1, ))
    assert_size_stride(arg445_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg446_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg447_1, (432, ), (1, ))
    assert_size_stride(arg448_1, (432, ), (1, ))
    assert_size_stride(arg449_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg450_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg451_1, (432, ), (1, ))
    assert_size_stride(arg452_1, (432, ), (1, ))
    assert_size_stride(arg453_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg454_1, (432, ), (1, ))
    assert_size_stride(arg455_1, (432, ), (1, ))
    assert_size_stride(arg456_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg457_1, (432, ), (1, ))
    assert_size_stride(arg458_1, (432, ), (1, ))
    assert_size_stride(arg459_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg460_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg461_1, (432, ), (1, ))
    assert_size_stride(arg462_1, (432, ), (1, ))
    assert_size_stride(arg463_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg464_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg465_1, (432, ), (1, ))
    assert_size_stride(arg466_1, (432, ), (1, ))
    assert_size_stride(arg467_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg468_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg469_1, (432, ), (1, ))
    assert_size_stride(arg470_1, (432, ), (1, ))
    assert_size_stride(arg471_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg472_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg473_1, (432, ), (1, ))
    assert_size_stride(arg474_1, (432, ), (1, ))
    assert_size_stride(arg475_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg476_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg477_1, (432, ), (1, ))
    assert_size_stride(arg478_1, (432, ), (1, ))
    assert_size_stride(arg479_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg480_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg481_1, (432, ), (1, ))
    assert_size_stride(arg482_1, (432, ), (1, ))
    assert_size_stride(arg483_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg484_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg485_1, (432, ), (1, ))
    assert_size_stride(arg486_1, (432, ), (1, ))
    assert_size_stride(arg487_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg488_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg489_1, (432, ), (1, ))
    assert_size_stride(arg490_1, (432, ), (1, ))
    assert_size_stride(arg491_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg492_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg493_1, (432, ), (1, ))
    assert_size_stride(arg494_1, (432, ), (1, ))
    assert_size_stride(arg495_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg496_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg497_1, (432, ), (1, ))
    assert_size_stride(arg498_1, (432, ), (1, ))
    assert_size_stride(arg499_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg500_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg501_1, (432, ), (1, ))
    assert_size_stride(arg502_1, (432, ), (1, ))
    assert_size_stride(arg503_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg504_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg505_1, (432, ), (1, ))
    assert_size_stride(arg506_1, (432, ), (1, ))
    assert_size_stride(arg507_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg508_1, (432, ), (1, ))
    assert_size_stride(arg509_1, (432, ), (1, ))
    assert_size_stride(arg510_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg511_1, (432, ), (1, ))
    assert_size_stride(arg512_1, (432, ), (1, ))
    assert_size_stride(arg513_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg514_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg515_1, (432, ), (1, ))
    assert_size_stride(arg516_1, (432, ), (1, ))
    assert_size_stride(arg517_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg518_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg519_1, (432, ), (1, ))
    assert_size_stride(arg520_1, (432, ), (1, ))
    assert_size_stride(arg521_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg522_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg523_1, (432, ), (1, ))
    assert_size_stride(arg524_1, (432, ), (1, ))
    assert_size_stride(arg525_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg526_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg527_1, (432, ), (1, ))
    assert_size_stride(arg528_1, (432, ), (1, ))
    assert_size_stride(arg529_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg530_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg531_1, (432, ), (1, ))
    assert_size_stride(arg532_1, (432, ), (1, ))
    assert_size_stride(arg533_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg534_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg535_1, (432, ), (1, ))
    assert_size_stride(arg536_1, (432, ), (1, ))
    assert_size_stride(arg537_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg538_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg539_1, (432, ), (1, ))
    assert_size_stride(arg540_1, (432, ), (1, ))
    assert_size_stride(arg541_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg542_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg543_1, (432, ), (1, ))
    assert_size_stride(arg544_1, (432, ), (1, ))
    assert_size_stride(arg545_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg546_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg547_1, (432, ), (1, ))
    assert_size_stride(arg548_1, (432, ), (1, ))
    assert_size_stride(arg549_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg550_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg551_1, (432, ), (1, ))
    assert_size_stride(arg552_1, (432, ), (1, ))
    assert_size_stride(arg553_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg554_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg555_1, (432, ), (1, ))
    assert_size_stride(arg556_1, (432, ), (1, ))
    assert_size_stride(arg557_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg558_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg559_1, (432, ), (1, ))
    assert_size_stride(arg560_1, (432, ), (1, ))
    assert_size_stride(arg561_1, (864, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg562_1, (864, ), (1, ))
    assert_size_stride(arg563_1, (864, ), (1, ))
    assert_size_stride(arg564_1, (864, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg565_1, (864, ), (1, ))
    assert_size_stride(arg566_1, (864, ), (1, ))
    assert_size_stride(arg567_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg568_1, (864, ), (1, ))
    assert_size_stride(arg569_1, (864, ), (1, ))
    assert_size_stride(arg570_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg571_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg572_1, (864, ), (1, ))
    assert_size_stride(arg573_1, (864, ), (1, ))
    assert_size_stride(arg574_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg575_1, (864, ), (1, ))
    assert_size_stride(arg576_1, (864, ), (1, ))
    assert_size_stride(arg577_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg578_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg579_1, (864, ), (1, ))
    assert_size_stride(arg580_1, (864, ), (1, ))
    assert_size_stride(arg581_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg582_1, (864, ), (1, ))
    assert_size_stride(arg583_1, (864, ), (1, ))
    assert_size_stride(arg584_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg585_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg586_1, (864, ), (1, ))
    assert_size_stride(arg587_1, (864, ), (1, ))
    assert_size_stride(arg588_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg589_1, (864, ), (1, ))
    assert_size_stride(arg590_1, (864, ), (1, ))
    assert_size_stride(arg591_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg592_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg593_1, (864, ), (1, ))
    assert_size_stride(arg594_1, (864, ), (1, ))
    assert_size_stride(arg595_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg596_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg597_1, (864, ), (1, ))
    assert_size_stride(arg598_1, (864, ), (1, ))
    assert_size_stride(arg599_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg600_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg601_1, (864, ), (1, ))
    assert_size_stride(arg602_1, (864, ), (1, ))
    assert_size_stride(arg603_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg604_1, (864, ), (1, ))
    assert_size_stride(arg605_1, (864, ), (1, ))
    assert_size_stride(arg606_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg607_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg608_1, (864, ), (1, ))
    assert_size_stride(arg609_1, (864, ), (1, ))
    assert_size_stride(arg610_1, (864, ), (1, ))
    assert_size_stride(arg611_1, (864, ), (1, ))
    assert_size_stride(arg612_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg613_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg614_1, (864, ), (1, ))
    assert_size_stride(arg615_1, (864, ), (1, ))
    assert_size_stride(arg616_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg617_1, (864, ), (1, ))
    assert_size_stride(arg618_1, (864, ), (1, ))
    assert_size_stride(arg619_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg620_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg621_1, (864, ), (1, ))
    assert_size_stride(arg622_1, (864, ), (1, ))
    assert_size_stride(arg623_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg624_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg625_1, (864, ), (1, ))
    assert_size_stride(arg626_1, (864, ), (1, ))
    assert_size_stride(arg627_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg628_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg629_1, (864, ), (1, ))
    assert_size_stride(arg630_1, (864, ), (1, ))
    assert_size_stride(arg631_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg632_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg633_1, (864, ), (1, ))
    assert_size_stride(arg634_1, (864, ), (1, ))
    assert_size_stride(arg635_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg636_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg637_1, (864, ), (1, ))
    assert_size_stride(arg638_1, (864, ), (1, ))
    assert_size_stride(arg639_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg640_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg641_1, (864, ), (1, ))
    assert_size_stride(arg642_1, (864, ), (1, ))
    assert_size_stride(arg643_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg644_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg645_1, (864, ), (1, ))
    assert_size_stride(arg646_1, (864, ), (1, ))
    assert_size_stride(arg647_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg648_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg649_1, (864, ), (1, ))
    assert_size_stride(arg650_1, (864, ), (1, ))
    assert_size_stride(arg651_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg652_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg653_1, (864, ), (1, ))
    assert_size_stride(arg654_1, (864, ), (1, ))
    assert_size_stride(arg655_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg656_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg657_1, (864, ), (1, ))
    assert_size_stride(arg658_1, (864, ), (1, ))
    assert_size_stride(arg659_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg660_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg661_1, (864, ), (1, ))
    assert_size_stride(arg662_1, (864, ), (1, ))
    assert_size_stride(arg663_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg664_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg665_1, (864, ), (1, ))
    assert_size_stride(arg666_1, (864, ), (1, ))
    assert_size_stride(arg667_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg668_1, (864, ), (1, ))
    assert_size_stride(arg669_1, (864, ), (1, ))
    assert_size_stride(arg670_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg671_1, (864, ), (1, ))
    assert_size_stride(arg672_1, (864, ), (1, ))
    assert_size_stride(arg673_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg674_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg675_1, (864, ), (1, ))
    assert_size_stride(arg676_1, (864, ), (1, ))
    assert_size_stride(arg677_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg678_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg679_1, (864, ), (1, ))
    assert_size_stride(arg680_1, (864, ), (1, ))
    assert_size_stride(arg681_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg682_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg683_1, (864, ), (1, ))
    assert_size_stride(arg684_1, (864, ), (1, ))
    assert_size_stride(arg685_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg686_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg687_1, (864, ), (1, ))
    assert_size_stride(arg688_1, (864, ), (1, ))
    assert_size_stride(arg689_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg690_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg691_1, (864, ), (1, ))
    assert_size_stride(arg692_1, (864, ), (1, ))
    assert_size_stride(arg693_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg694_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg695_1, (864, ), (1, ))
    assert_size_stride(arg696_1, (864, ), (1, ))
    assert_size_stride(arg697_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg698_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg699_1, (864, ), (1, ))
    assert_size_stride(arg700_1, (864, ), (1, ))
    assert_size_stride(arg701_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg702_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg703_1, (864, ), (1, ))
    assert_size_stride(arg704_1, (864, ), (1, ))
    assert_size_stride(arg705_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg706_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg707_1, (864, ), (1, ))
    assert_size_stride(arg708_1, (864, ), (1, ))
    assert_size_stride(arg709_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg710_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg711_1, (864, ), (1, ))
    assert_size_stride(arg712_1, (864, ), (1, ))
    assert_size_stride(arg713_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg714_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg715_1, (864, ), (1, ))
    assert_size_stride(arg716_1, (864, ), (1, ))
    assert_size_stride(arg717_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg718_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg719_1, (864, ), (1, ))
    assert_size_stride(arg720_1, (864, ), (1, ))
    assert_size_stride(arg721_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg722_1, (864, ), (1, ))
    assert_size_stride(arg723_1, (864, ), (1, ))
    assert_size_stride(arg724_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg725_1, (864, ), (1, ))
    assert_size_stride(arg726_1, (864, ), (1, ))
    assert_size_stride(arg727_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg728_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg729_1, (864, ), (1, ))
    assert_size_stride(arg730_1, (864, ), (1, ))
    assert_size_stride(arg731_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg732_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg733_1, (864, ), (1, ))
    assert_size_stride(arg734_1, (864, ), (1, ))
    assert_size_stride(arg735_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg736_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg737_1, (864, ), (1, ))
    assert_size_stride(arg738_1, (864, ), (1, ))
    assert_size_stride(arg739_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg740_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg741_1, (864, ), (1, ))
    assert_size_stride(arg742_1, (864, ), (1, ))
    assert_size_stride(arg743_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg744_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg745_1, (864, ), (1, ))
    assert_size_stride(arg746_1, (864, ), (1, ))
    assert_size_stride(arg747_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg748_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg749_1, (864, ), (1, ))
    assert_size_stride(arg750_1, (864, ), (1, ))
    assert_size_stride(arg751_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg752_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg753_1, (864, ), (1, ))
    assert_size_stride(arg754_1, (864, ), (1, ))
    assert_size_stride(arg755_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg756_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg757_1, (864, ), (1, ))
    assert_size_stride(arg758_1, (864, ), (1, ))
    assert_size_stride(arg759_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg760_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg761_1, (864, ), (1, ))
    assert_size_stride(arg762_1, (864, ), (1, ))
    assert_size_stride(arg763_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg764_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg765_1, (864, ), (1, ))
    assert_size_stride(arg766_1, (864, ), (1, ))
    assert_size_stride(arg767_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg768_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg769_1, (864, ), (1, ))
    assert_size_stride(arg770_1, (864, ), (1, ))
    assert_size_stride(arg771_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg772_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg773_1, (864, ), (1, ))
    assert_size_stride(arg774_1, (864, ), (1, ))
    assert_size_stride(arg775_1, (1000, 4320), (4320, 1))
    assert_size_stride(arg776_1, (1000, ), (1, ))
    assert_size_stride(arg777_1, (96, ), (1, ))
    assert_size_stride(arg778_1, (96, ), (1, ))
    assert_size_stride(arg779_1, (54, ), (1, ))
    assert_size_stride(arg780_1, (54, ), (1, ))
    assert_size_stride(arg781_1, (), ())
    assert_size_stride(arg782_1, (54, ), (1, ))
    assert_size_stride(arg783_1, (54, ), (1, ))
    assert_size_stride(arg784_1, (), ())
    assert_size_stride(arg785_1, (54, ), (1, ))
    assert_size_stride(arg786_1, (54, ), (1, ))
    assert_size_stride(arg787_1, (), ())
    assert_size_stride(arg788_1, (54, ), (1, ))
    assert_size_stride(arg789_1, (54, ), (1, ))
    assert_size_stride(arg790_1, (), ())
    assert_size_stride(arg791_1, (54, ), (1, ))
    assert_size_stride(arg792_1, (54, ), (1, ))
    assert_size_stride(arg793_1, (), ())
    assert_size_stride(arg794_1, (54, ), (1, ))
    assert_size_stride(arg795_1, (54, ), (1, ))
    assert_size_stride(arg796_1, (), ())
    assert_size_stride(arg797_1, (54, ), (1, ))
    assert_size_stride(arg798_1, (54, ), (1, ))
    assert_size_stride(arg799_1, (), ())
    assert_size_stride(arg800_1, (54, ), (1, ))
    assert_size_stride(arg801_1, (54, ), (1, ))
    assert_size_stride(arg802_1, (), ())
    assert_size_stride(arg803_1, (54, ), (1, ))
    assert_size_stride(arg804_1, (54, ), (1, ))
    assert_size_stride(arg805_1, (), ())
    assert_size_stride(arg806_1, (54, ), (1, ))
    assert_size_stride(arg807_1, (54, ), (1, ))
    assert_size_stride(arg808_1, (), ())
    assert_size_stride(arg809_1, (54, ), (1, ))
    assert_size_stride(arg810_1, (54, ), (1, ))
    assert_size_stride(arg811_1, (), ())
    assert_size_stride(arg812_1, (54, ), (1, ))
    assert_size_stride(arg813_1, (54, ), (1, ))
    assert_size_stride(arg814_1, (), ())
    assert_size_stride(arg815_1, (54, ), (1, ))
    assert_size_stride(arg816_1, (54, ), (1, ))
    assert_size_stride(arg817_1, (), ())
    assert_size_stride(arg818_1, (54, ), (1, ))
    assert_size_stride(arg819_1, (54, ), (1, ))
    assert_size_stride(arg820_1, (), ())
    assert_size_stride(arg821_1, (54, ), (1, ))
    assert_size_stride(arg822_1, (54, ), (1, ))
    assert_size_stride(arg823_1, (), ())
    assert_size_stride(arg824_1, (108, ), (1, ))
    assert_size_stride(arg825_1, (108, ), (1, ))
    assert_size_stride(arg826_1, (), ())
    assert_size_stride(arg827_1, (108, ), (1, ))
    assert_size_stride(arg828_1, (108, ), (1, ))
    assert_size_stride(arg829_1, (), ())
    assert_size_stride(arg830_1, (108, ), (1, ))
    assert_size_stride(arg831_1, (108, ), (1, ))
    assert_size_stride(arg832_1, (), ())
    assert_size_stride(arg833_1, (108, ), (1, ))
    assert_size_stride(arg834_1, (108, ), (1, ))
    assert_size_stride(arg835_1, (), ())
    assert_size_stride(arg836_1, (108, ), (1, ))
    assert_size_stride(arg837_1, (108, ), (1, ))
    assert_size_stride(arg838_1, (), ())
    assert_size_stride(arg839_1, (108, ), (1, ))
    assert_size_stride(arg840_1, (108, ), (1, ))
    assert_size_stride(arg841_1, (), ())
    assert_size_stride(arg842_1, (108, ), (1, ))
    assert_size_stride(arg843_1, (108, ), (1, ))
    assert_size_stride(arg844_1, (), ())
    assert_size_stride(arg845_1, (108, ), (1, ))
    assert_size_stride(arg846_1, (108, ), (1, ))
    assert_size_stride(arg847_1, (), ())
    assert_size_stride(arg848_1, (108, ), (1, ))
    assert_size_stride(arg849_1, (108, ), (1, ))
    assert_size_stride(arg850_1, (), ())
    assert_size_stride(arg851_1, (108, ), (1, ))
    assert_size_stride(arg852_1, (108, ), (1, ))
    assert_size_stride(arg853_1, (), ())
    assert_size_stride(arg854_1, (108, ), (1, ))
    assert_size_stride(arg855_1, (108, ), (1, ))
    assert_size_stride(arg856_1, (), ())
    assert_size_stride(arg857_1, (108, ), (1, ))
    assert_size_stride(arg858_1, (108, ), (1, ))
    assert_size_stride(arg859_1, (), ())
    assert_size_stride(arg860_1, (108, ), (1, ))
    assert_size_stride(arg861_1, (108, ), (1, ))
    assert_size_stride(arg862_1, (), ())
    assert_size_stride(arg863_1, (108, ), (1, ))
    assert_size_stride(arg864_1, (108, ), (1, ))
    assert_size_stride(arg865_1, (), ())
    assert_size_stride(arg866_1, (108, ), (1, ))
    assert_size_stride(arg867_1, (108, ), (1, ))
    assert_size_stride(arg868_1, (), ())
    assert_size_stride(arg869_1, (216, ), (1, ))
    assert_size_stride(arg870_1, (216, ), (1, ))
    assert_size_stride(arg871_1, (), ())
    assert_size_stride(arg872_1, (216, ), (1, ))
    assert_size_stride(arg873_1, (216, ), (1, ))
    assert_size_stride(arg874_1, (), ())
    assert_size_stride(arg875_1, (216, ), (1, ))
    assert_size_stride(arg876_1, (216, ), (1, ))
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (216, ), (1, ))
    assert_size_stride(arg879_1, (216, ), (1, ))
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (216, ), (1, ))
    assert_size_stride(arg882_1, (216, ), (1, ))
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (216, ), (1, ))
    assert_size_stride(arg885_1, (216, ), (1, ))
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (216, ), (1, ))
    assert_size_stride(arg888_1, (216, ), (1, ))
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (216, ), (1, ))
    assert_size_stride(arg891_1, (216, ), (1, ))
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (216, ), (1, ))
    assert_size_stride(arg894_1, (216, ), (1, ))
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (216, ), (1, ))
    assert_size_stride(arg897_1, (216, ), (1, ))
    assert_size_stride(arg898_1, (), ())
    assert_size_stride(arg899_1, (216, ), (1, ))
    assert_size_stride(arg900_1, (216, ), (1, ))
    assert_size_stride(arg901_1, (), ())
    assert_size_stride(arg902_1, (216, ), (1, ))
    assert_size_stride(arg903_1, (216, ), (1, ))
    assert_size_stride(arg904_1, (), ())
    assert_size_stride(arg905_1, (216, ), (1, ))
    assert_size_stride(arg906_1, (216, ), (1, ))
    assert_size_stride(arg907_1, (), ())
    assert_size_stride(arg908_1, (216, ), (1, ))
    assert_size_stride(arg909_1, (216, ), (1, ))
    assert_size_stride(arg910_1, (), ())
    assert_size_stride(arg911_1, (216, ), (1, ))
    assert_size_stride(arg912_1, (216, ), (1, ))
    assert_size_stride(arg913_1, (), ())
    assert_size_stride(arg914_1, (216, ), (1, ))
    assert_size_stride(arg915_1, (216, ), (1, ))
    assert_size_stride(arg916_1, (), ())
    assert_size_stride(arg917_1, (216, ), (1, ))
    assert_size_stride(arg918_1, (216, ), (1, ))
    assert_size_stride(arg919_1, (), ())
    assert_size_stride(arg920_1, (216, ), (1, ))
    assert_size_stride(arg921_1, (216, ), (1, ))
    assert_size_stride(arg922_1, (), ())
    assert_size_stride(arg923_1, (216, ), (1, ))
    assert_size_stride(arg924_1, (216, ), (1, ))
    assert_size_stride(arg925_1, (), ())
    assert_size_stride(arg926_1, (216, ), (1, ))
    assert_size_stride(arg927_1, (216, ), (1, ))
    assert_size_stride(arg928_1, (), ())
    assert_size_stride(arg929_1, (216, ), (1, ))
    assert_size_stride(arg930_1, (216, ), (1, ))
    assert_size_stride(arg931_1, (), ())
    assert_size_stride(arg932_1, (216, ), (1, ))
    assert_size_stride(arg933_1, (216, ), (1, ))
    assert_size_stride(arg934_1, (), ())
    assert_size_stride(arg935_1, (216, ), (1, ))
    assert_size_stride(arg936_1, (216, ), (1, ))
    assert_size_stride(arg937_1, (), ())
    assert_size_stride(arg938_1, (216, ), (1, ))
    assert_size_stride(arg939_1, (216, ), (1, ))
    assert_size_stride(arg940_1, (), ())
    assert_size_stride(arg941_1, (216, ), (1, ))
    assert_size_stride(arg942_1, (216, ), (1, ))
    assert_size_stride(arg943_1, (), ())
    assert_size_stride(arg944_1, (216, ), (1, ))
    assert_size_stride(arg945_1, (216, ), (1, ))
    assert_size_stride(arg946_1, (), ())
    assert_size_stride(arg947_1, (216, ), (1, ))
    assert_size_stride(arg948_1, (216, ), (1, ))
    assert_size_stride(arg949_1, (), ())
    assert_size_stride(arg950_1, (216, ), (1, ))
    assert_size_stride(arg951_1, (216, ), (1, ))
    assert_size_stride(arg952_1, (), ())
    assert_size_stride(arg953_1, (216, ), (1, ))
    assert_size_stride(arg954_1, (216, ), (1, ))
    assert_size_stride(arg955_1, (), ())
    assert_size_stride(arg956_1, (216, ), (1, ))
    assert_size_stride(arg957_1, (216, ), (1, ))
    assert_size_stride(arg958_1, (), ())
    assert_size_stride(arg959_1, (216, ), (1, ))
    assert_size_stride(arg960_1, (216, ), (1, ))
    assert_size_stride(arg961_1, (), ())
    assert_size_stride(arg962_1, (216, ), (1, ))
    assert_size_stride(arg963_1, (216, ), (1, ))
    assert_size_stride(arg964_1, (), ())
    assert_size_stride(arg965_1, (216, ), (1, ))
    assert_size_stride(arg966_1, (216, ), (1, ))
    assert_size_stride(arg967_1, (), ())
    assert_size_stride(arg968_1, (216, ), (1, ))
    assert_size_stride(arg969_1, (216, ), (1, ))
    assert_size_stride(arg970_1, (), ())
    assert_size_stride(arg971_1, (216, ), (1, ))
    assert_size_stride(arg972_1, (216, ), (1, ))
    assert_size_stride(arg973_1, (), ())
    assert_size_stride(arg974_1, (216, ), (1, ))
    assert_size_stride(arg975_1, (216, ), (1, ))
    assert_size_stride(arg976_1, (), ())
    assert_size_stride(arg977_1, (216, ), (1, ))
    assert_size_stride(arg978_1, (216, ), (1, ))
    assert_size_stride(arg979_1, (), ())
    assert_size_stride(arg980_1, (216, ), (1, ))
    assert_size_stride(arg981_1, (216, ), (1, ))
    assert_size_stride(arg982_1, (), ())
    assert_size_stride(arg983_1, (216, ), (1, ))
    assert_size_stride(arg984_1, (216, ), (1, ))
    assert_size_stride(arg985_1, (), ())
    assert_size_stride(arg986_1, (216, ), (1, ))
    assert_size_stride(arg987_1, (216, ), (1, ))
    assert_size_stride(arg988_1, (), ())
    assert_size_stride(arg989_1, (216, ), (1, ))
    assert_size_stride(arg990_1, (216, ), (1, ))
    assert_size_stride(arg991_1, (), ())
    assert_size_stride(arg992_1, (216, ), (1, ))
    assert_size_stride(arg993_1, (216, ), (1, ))
    assert_size_stride(arg994_1, (), ())
    assert_size_stride(arg995_1, (216, ), (1, ))
    assert_size_stride(arg996_1, (216, ), (1, ))
    assert_size_stride(arg997_1, (), ())
    assert_size_stride(arg998_1, (216, ), (1, ))
    assert_size_stride(arg999_1, (216, ), (1, ))
    assert_size_stride(arg1000_1, (), ())
    assert_size_stride(arg1001_1, (216, ), (1, ))
    assert_size_stride(arg1002_1, (216, ), (1, ))
    assert_size_stride(arg1003_1, (), ())
    assert_size_stride(arg1004_1, (216, ), (1, ))
    assert_size_stride(arg1005_1, (216, ), (1, ))
    assert_size_stride(arg1006_1, (), ())
    assert_size_stride(arg1007_1, (216, ), (1, ))
    assert_size_stride(arg1008_1, (216, ), (1, ))
    assert_size_stride(arg1009_1, (), ())
    assert_size_stride(arg1010_1, (216, ), (1, ))
    assert_size_stride(arg1011_1, (216, ), (1, ))
    assert_size_stride(arg1012_1, (), ())
    assert_size_stride(arg1013_1, (216, ), (1, ))
    assert_size_stride(arg1014_1, (216, ), (1, ))
    assert_size_stride(arg1015_1, (), ())
    assert_size_stride(arg1016_1, (216, ), (1, ))
    assert_size_stride(arg1017_1, (216, ), (1, ))
    assert_size_stride(arg1018_1, (), ())
    assert_size_stride(arg1019_1, (216, ), (1, ))
    assert_size_stride(arg1020_1, (216, ), (1, ))
    assert_size_stride(arg1021_1, (), ())
    assert_size_stride(arg1022_1, (216, ), (1, ))
    assert_size_stride(arg1023_1, (216, ), (1, ))
    assert_size_stride(arg1024_1, (), ())
    assert_size_stride(arg1025_1, (216, ), (1, ))
    assert_size_stride(arg1026_1, (216, ), (1, ))
    assert_size_stride(arg1027_1, (), ())
    assert_size_stride(arg1028_1, (216, ), (1, ))
    assert_size_stride(arg1029_1, (216, ), (1, ))
    assert_size_stride(arg1030_1, (), ())
    assert_size_stride(arg1031_1, (216, ), (1, ))
    assert_size_stride(arg1032_1, (216, ), (1, ))
    assert_size_stride(arg1033_1, (), ())
    assert_size_stride(arg1034_1, (216, ), (1, ))
    assert_size_stride(arg1035_1, (216, ), (1, ))
    assert_size_stride(arg1036_1, (), ())
    assert_size_stride(arg1037_1, (432, ), (1, ))
    assert_size_stride(arg1038_1, (432, ), (1, ))
    assert_size_stride(arg1039_1, (), ())
    assert_size_stride(arg1040_1, (432, ), (1, ))
    assert_size_stride(arg1041_1, (432, ), (1, ))
    assert_size_stride(arg1042_1, (), ())
    assert_size_stride(arg1043_1, (432, ), (1, ))
    assert_size_stride(arg1044_1, (432, ), (1, ))
    assert_size_stride(arg1045_1, (), ())
    assert_size_stride(arg1046_1, (432, ), (1, ))
    assert_size_stride(arg1047_1, (432, ), (1, ))
    assert_size_stride(arg1048_1, (), ())
    assert_size_stride(arg1049_1, (432, ), (1, ))
    assert_size_stride(arg1050_1, (432, ), (1, ))
    assert_size_stride(arg1051_1, (), ())
    assert_size_stride(arg1052_1, (432, ), (1, ))
    assert_size_stride(arg1053_1, (432, ), (1, ))
    assert_size_stride(arg1054_1, (), ())
    assert_size_stride(arg1055_1, (432, ), (1, ))
    assert_size_stride(arg1056_1, (432, ), (1, ))
    assert_size_stride(arg1057_1, (), ())
    assert_size_stride(arg1058_1, (432, ), (1, ))
    assert_size_stride(arg1059_1, (432, ), (1, ))
    assert_size_stride(arg1060_1, (), ())
    assert_size_stride(arg1061_1, (432, ), (1, ))
    assert_size_stride(arg1062_1, (432, ), (1, ))
    assert_size_stride(arg1063_1, (), ())
    assert_size_stride(arg1064_1, (432, ), (1, ))
    assert_size_stride(arg1065_1, (432, ), (1, ))
    assert_size_stride(arg1066_1, (), ())
    assert_size_stride(arg1067_1, (432, ), (1, ))
    assert_size_stride(arg1068_1, (432, ), (1, ))
    assert_size_stride(arg1069_1, (), ())
    assert_size_stride(arg1070_1, (432, ), (1, ))
    assert_size_stride(arg1071_1, (432, ), (1, ))
    assert_size_stride(arg1072_1, (), ())
    assert_size_stride(arg1073_1, (432, ), (1, ))
    assert_size_stride(arg1074_1, (432, ), (1, ))
    assert_size_stride(arg1075_1, (), ())
    assert_size_stride(arg1076_1, (432, ), (1, ))
    assert_size_stride(arg1077_1, (432, ), (1, ))
    assert_size_stride(arg1078_1, (), ())
    assert_size_stride(arg1079_1, (432, ), (1, ))
    assert_size_stride(arg1080_1, (432, ), (1, ))
    assert_size_stride(arg1081_1, (), ())
    assert_size_stride(arg1082_1, (432, ), (1, ))
    assert_size_stride(arg1083_1, (432, ), (1, ))
    assert_size_stride(arg1084_1, (), ())
    assert_size_stride(arg1085_1, (432, ), (1, ))
    assert_size_stride(arg1086_1, (432, ), (1, ))
    assert_size_stride(arg1087_1, (), ())
    assert_size_stride(arg1088_1, (432, ), (1, ))
    assert_size_stride(arg1089_1, (432, ), (1, ))
    assert_size_stride(arg1090_1, (), ())
    assert_size_stride(arg1091_1, (432, ), (1, ))
    assert_size_stride(arg1092_1, (432, ), (1, ))
    assert_size_stride(arg1093_1, (), ())
    assert_size_stride(arg1094_1, (432, ), (1, ))
    assert_size_stride(arg1095_1, (432, ), (1, ))
    assert_size_stride(arg1096_1, (), ())
    assert_size_stride(arg1097_1, (432, ), (1, ))
    assert_size_stride(arg1098_1, (432, ), (1, ))
    assert_size_stride(arg1099_1, (), ())
    assert_size_stride(arg1100_1, (432, ), (1, ))
    assert_size_stride(arg1101_1, (432, ), (1, ))
    assert_size_stride(arg1102_1, (), ())
    assert_size_stride(arg1103_1, (432, ), (1, ))
    assert_size_stride(arg1104_1, (432, ), (1, ))
    assert_size_stride(arg1105_1, (), ())
    assert_size_stride(arg1106_1, (432, ), (1, ))
    assert_size_stride(arg1107_1, (432, ), (1, ))
    assert_size_stride(arg1108_1, (), ())
    assert_size_stride(arg1109_1, (432, ), (1, ))
    assert_size_stride(arg1110_1, (432, ), (1, ))
    assert_size_stride(arg1111_1, (), ())
    assert_size_stride(arg1112_1, (432, ), (1, ))
    assert_size_stride(arg1113_1, (432, ), (1, ))
    assert_size_stride(arg1114_1, (), ())
    assert_size_stride(arg1115_1, (432, ), (1, ))
    assert_size_stride(arg1116_1, (432, ), (1, ))
    assert_size_stride(arg1117_1, (), ())
    assert_size_stride(arg1118_1, (432, ), (1, ))
    assert_size_stride(arg1119_1, (432, ), (1, ))
    assert_size_stride(arg1120_1, (), ())
    assert_size_stride(arg1121_1, (432, ), (1, ))
    assert_size_stride(arg1122_1, (432, ), (1, ))
    assert_size_stride(arg1123_1, (), ())
    assert_size_stride(arg1124_1, (432, ), (1, ))
    assert_size_stride(arg1125_1, (432, ), (1, ))
    assert_size_stride(arg1126_1, (), ())
    assert_size_stride(arg1127_1, (432, ), (1, ))
    assert_size_stride(arg1128_1, (432, ), (1, ))
    assert_size_stride(arg1129_1, (), ())
    assert_size_stride(arg1130_1, (432, ), (1, ))
    assert_size_stride(arg1131_1, (432, ), (1, ))
    assert_size_stride(arg1132_1, (), ())
    assert_size_stride(arg1133_1, (432, ), (1, ))
    assert_size_stride(arg1134_1, (432, ), (1, ))
    assert_size_stride(arg1135_1, (), ())
    assert_size_stride(arg1136_1, (432, ), (1, ))
    assert_size_stride(arg1137_1, (432, ), (1, ))
    assert_size_stride(arg1138_1, (), ())
    assert_size_stride(arg1139_1, (432, ), (1, ))
    assert_size_stride(arg1140_1, (432, ), (1, ))
    assert_size_stride(arg1141_1, (), ())
    assert_size_stride(arg1142_1, (432, ), (1, ))
    assert_size_stride(arg1143_1, (432, ), (1, ))
    assert_size_stride(arg1144_1, (), ())
    assert_size_stride(arg1145_1, (432, ), (1, ))
    assert_size_stride(arg1146_1, (432, ), (1, ))
    assert_size_stride(arg1147_1, (), ())
    assert_size_stride(arg1148_1, (432, ), (1, ))
    assert_size_stride(arg1149_1, (432, ), (1, ))
    assert_size_stride(arg1150_1, (), ())
    assert_size_stride(arg1151_1, (432, ), (1, ))
    assert_size_stride(arg1152_1, (432, ), (1, ))
    assert_size_stride(arg1153_1, (), ())
    assert_size_stride(arg1154_1, (432, ), (1, ))
    assert_size_stride(arg1155_1, (432, ), (1, ))
    assert_size_stride(arg1156_1, (), ())
    assert_size_stride(arg1157_1, (432, ), (1, ))
    assert_size_stride(arg1158_1, (432, ), (1, ))
    assert_size_stride(arg1159_1, (), ())
    assert_size_stride(arg1160_1, (432, ), (1, ))
    assert_size_stride(arg1161_1, (432, ), (1, ))
    assert_size_stride(arg1162_1, (), ())
    assert_size_stride(arg1163_1, (432, ), (1, ))
    assert_size_stride(arg1164_1, (432, ), (1, ))
    assert_size_stride(arg1165_1, (), ())
    assert_size_stride(arg1166_1, (432, ), (1, ))
    assert_size_stride(arg1167_1, (432, ), (1, ))
    assert_size_stride(arg1168_1, (), ())
    assert_size_stride(arg1169_1, (432, ), (1, ))
    assert_size_stride(arg1170_1, (432, ), (1, ))
    assert_size_stride(arg1171_1, (), ())
    assert_size_stride(arg1172_1, (432, ), (1, ))
    assert_size_stride(arg1173_1, (432, ), (1, ))
    assert_size_stride(arg1174_1, (), ())
    assert_size_stride(arg1175_1, (432, ), (1, ))
    assert_size_stride(arg1176_1, (432, ), (1, ))
    assert_size_stride(arg1177_1, (), ())
    assert_size_stride(arg1178_1, (432, ), (1, ))
    assert_size_stride(arg1179_1, (432, ), (1, ))
    assert_size_stride(arg1180_1, (), ())
    assert_size_stride(arg1181_1, (432, ), (1, ))
    assert_size_stride(arg1182_1, (432, ), (1, ))
    assert_size_stride(arg1183_1, (), ())
    assert_size_stride(arg1184_1, (432, ), (1, ))
    assert_size_stride(arg1185_1, (432, ), (1, ))
    assert_size_stride(arg1186_1, (), ())
    assert_size_stride(arg1187_1, (432, ), (1, ))
    assert_size_stride(arg1188_1, (432, ), (1, ))
    assert_size_stride(arg1189_1, (), ())
    assert_size_stride(arg1190_1, (432, ), (1, ))
    assert_size_stride(arg1191_1, (432, ), (1, ))
    assert_size_stride(arg1192_1, (), ())
    assert_size_stride(arg1193_1, (432, ), (1, ))
    assert_size_stride(arg1194_1, (432, ), (1, ))
    assert_size_stride(arg1195_1, (), ())
    assert_size_stride(arg1196_1, (432, ), (1, ))
    assert_size_stride(arg1197_1, (432, ), (1, ))
    assert_size_stride(arg1198_1, (), ())
    assert_size_stride(arg1199_1, (432, ), (1, ))
    assert_size_stride(arg1200_1, (432, ), (1, ))
    assert_size_stride(arg1201_1, (), ())
    assert_size_stride(arg1202_1, (432, ), (1, ))
    assert_size_stride(arg1203_1, (432, ), (1, ))
    assert_size_stride(arg1204_1, (), ())
    assert_size_stride(arg1205_1, (432, ), (1, ))
    assert_size_stride(arg1206_1, (432, ), (1, ))
    assert_size_stride(arg1207_1, (), ())
    assert_size_stride(arg1208_1, (864, ), (1, ))
    assert_size_stride(arg1209_1, (864, ), (1, ))
    assert_size_stride(arg1210_1, (), ())
    assert_size_stride(arg1211_1, (864, ), (1, ))
    assert_size_stride(arg1212_1, (864, ), (1, ))
    assert_size_stride(arg1213_1, (), ())
    assert_size_stride(arg1214_1, (864, ), (1, ))
    assert_size_stride(arg1215_1, (864, ), (1, ))
    assert_size_stride(arg1216_1, (), ())
    assert_size_stride(arg1217_1, (864, ), (1, ))
    assert_size_stride(arg1218_1, (864, ), (1, ))
    assert_size_stride(arg1219_1, (), ())
    assert_size_stride(arg1220_1, (864, ), (1, ))
    assert_size_stride(arg1221_1, (864, ), (1, ))
    assert_size_stride(arg1222_1, (), ())
    assert_size_stride(arg1223_1, (864, ), (1, ))
    assert_size_stride(arg1224_1, (864, ), (1, ))
    assert_size_stride(arg1225_1, (), ())
    assert_size_stride(arg1226_1, (864, ), (1, ))
    assert_size_stride(arg1227_1, (864, ), (1, ))
    assert_size_stride(arg1228_1, (), ())
    assert_size_stride(arg1229_1, (864, ), (1, ))
    assert_size_stride(arg1230_1, (864, ), (1, ))
    assert_size_stride(arg1231_1, (), ())
    assert_size_stride(arg1232_1, (864, ), (1, ))
    assert_size_stride(arg1233_1, (864, ), (1, ))
    assert_size_stride(arg1234_1, (), ())
    assert_size_stride(arg1235_1, (864, ), (1, ))
    assert_size_stride(arg1236_1, (864, ), (1, ))
    assert_size_stride(arg1237_1, (), ())
    assert_size_stride(arg1238_1, (864, ), (1, ))
    assert_size_stride(arg1239_1, (864, ), (1, ))
    assert_size_stride(arg1240_1, (), ())
    assert_size_stride(arg1241_1, (864, ), (1, ))
    assert_size_stride(arg1242_1, (864, ), (1, ))
    assert_size_stride(arg1243_1, (), ())
    assert_size_stride(arg1244_1, (864, ), (1, ))
    assert_size_stride(arg1245_1, (864, ), (1, ))
    assert_size_stride(arg1246_1, (), ())
    assert_size_stride(arg1247_1, (864, ), (1, ))
    assert_size_stride(arg1248_1, (864, ), (1, ))
    assert_size_stride(arg1249_1, (), ())
    assert_size_stride(arg1250_1, (864, ), (1, ))
    assert_size_stride(arg1251_1, (864, ), (1, ))
    assert_size_stride(arg1252_1, (), ())
    assert_size_stride(arg1253_1, (864, ), (1, ))
    assert_size_stride(arg1254_1, (864, ), (1, ))
    assert_size_stride(arg1255_1, (), ())
    assert_size_stride(arg1256_1, (864, ), (1, ))
    assert_size_stride(arg1257_1, (864, ), (1, ))
    assert_size_stride(arg1258_1, (), ())
    assert_size_stride(arg1259_1, (864, ), (1, ))
    assert_size_stride(arg1260_1, (864, ), (1, ))
    assert_size_stride(arg1261_1, (), ())
    assert_size_stride(arg1262_1, (864, ), (1, ))
    assert_size_stride(arg1263_1, (864, ), (1, ))
    assert_size_stride(arg1264_1, (), ())
    assert_size_stride(arg1265_1, (864, ), (1, ))
    assert_size_stride(arg1266_1, (864, ), (1, ))
    assert_size_stride(arg1267_1, (), ())
    assert_size_stride(arg1268_1, (864, ), (1, ))
    assert_size_stride(arg1269_1, (864, ), (1, ))
    assert_size_stride(arg1270_1, (), ())
    assert_size_stride(arg1271_1, (864, ), (1, ))
    assert_size_stride(arg1272_1, (864, ), (1, ))
    assert_size_stride(arg1273_1, (), ())
    assert_size_stride(arg1274_1, (864, ), (1, ))
    assert_size_stride(arg1275_1, (864, ), (1, ))
    assert_size_stride(arg1276_1, (), ())
    assert_size_stride(arg1277_1, (864, ), (1, ))
    assert_size_stride(arg1278_1, (864, ), (1, ))
    assert_size_stride(arg1279_1, (), ())
    assert_size_stride(arg1280_1, (864, ), (1, ))
    assert_size_stride(arg1281_1, (864, ), (1, ))
    assert_size_stride(arg1282_1, (), ())
    assert_size_stride(arg1283_1, (864, ), (1, ))
    assert_size_stride(arg1284_1, (864, ), (1, ))
    assert_size_stride(arg1285_1, (), ())
    assert_size_stride(arg1286_1, (864, ), (1, ))
    assert_size_stride(arg1287_1, (864, ), (1, ))
    assert_size_stride(arg1288_1, (), ())
    assert_size_stride(arg1289_1, (864, ), (1, ))
    assert_size_stride(arg1290_1, (864, ), (1, ))
    assert_size_stride(arg1291_1, (), ())
    assert_size_stride(arg1292_1, (864, ), (1, ))
    assert_size_stride(arg1293_1, (864, ), (1, ))
    assert_size_stride(arg1294_1, (), ())
    assert_size_stride(arg1295_1, (864, ), (1, ))
    assert_size_stride(arg1296_1, (864, ), (1, ))
    assert_size_stride(arg1297_1, (), ())
    assert_size_stride(arg1298_1, (864, ), (1, ))
    assert_size_stride(arg1299_1, (864, ), (1, ))
    assert_size_stride(arg1300_1, (), ())
    assert_size_stride(arg1301_1, (864, ), (1, ))
    assert_size_stride(arg1302_1, (864, ), (1, ))
    assert_size_stride(arg1303_1, (), ())
    assert_size_stride(arg1304_1, (864, ), (1, ))
    assert_size_stride(arg1305_1, (864, ), (1, ))
    assert_size_stride(arg1306_1, (), ())
    assert_size_stride(arg1307_1, (864, ), (1, ))
    assert_size_stride(arg1308_1, (864, ), (1, ))
    assert_size_stride(arg1309_1, (), ())
    assert_size_stride(arg1310_1, (864, ), (1, ))
    assert_size_stride(arg1311_1, (864, ), (1, ))
    assert_size_stride(arg1312_1, (), ())
    assert_size_stride(arg1313_1, (864, ), (1, ))
    assert_size_stride(arg1314_1, (864, ), (1, ))
    assert_size_stride(arg1315_1, (), ())
    assert_size_stride(arg1316_1, (864, ), (1, ))
    assert_size_stride(arg1317_1, (864, ), (1, ))
    assert_size_stride(arg1318_1, (), ())
    assert_size_stride(arg1319_1, (864, ), (1, ))
    assert_size_stride(arg1320_1, (864, ), (1, ))
    assert_size_stride(arg1321_1, (), ())
    assert_size_stride(arg1322_1, (864, ), (1, ))
    assert_size_stride(arg1323_1, (864, ), (1, ))
    assert_size_stride(arg1324_1, (), ())
    assert_size_stride(arg1325_1, (864, ), (1, ))
    assert_size_stride(arg1326_1, (864, ), (1, ))
    assert_size_stride(arg1327_1, (), ())
    assert_size_stride(arg1328_1, (864, ), (1, ))
    assert_size_stride(arg1329_1, (864, ), (1, ))
    assert_size_stride(arg1330_1, (), ())
    assert_size_stride(arg1331_1, (864, ), (1, ))
    assert_size_stride(arg1332_1, (864, ), (1, ))
    assert_size_stride(arg1333_1, (), ())
    assert_size_stride(arg1334_1, (864, ), (1, ))
    assert_size_stride(arg1335_1, (864, ), (1, ))
    assert_size_stride(arg1336_1, (), ())
    assert_size_stride(arg1337_1, (864, ), (1, ))
    assert_size_stride(arg1338_1, (864, ), (1, ))
    assert_size_stride(arg1339_1, (), ())
    assert_size_stride(arg1340_1, (864, ), (1, ))
    assert_size_stride(arg1341_1, (864, ), (1, ))
    assert_size_stride(arg1342_1, (), ())
    assert_size_stride(arg1343_1, (864, ), (1, ))
    assert_size_stride(arg1344_1, (864, ), (1, ))
    assert_size_stride(arg1345_1, (), ())
    assert_size_stride(arg1346_1, (864, ), (1, ))
    assert_size_stride(arg1347_1, (864, ), (1, ))
    assert_size_stride(arg1348_1, (), ())
    assert_size_stride(arg1349_1, (864, ), (1, ))
    assert_size_stride(arg1350_1, (864, ), (1, ))
    assert_size_stride(arg1351_1, (), ())
    assert_size_stride(arg1352_1, (864, ), (1, ))
    assert_size_stride(arg1353_1, (864, ), (1, ))
    assert_size_stride(arg1354_1, (), ())
    assert_size_stride(arg1355_1, (864, ), (1, ))
    assert_size_stride(arg1356_1, (864, ), (1, ))
    assert_size_stride(arg1357_1, (), ())
    assert_size_stride(arg1358_1, (864, ), (1, ))
    assert_size_stride(arg1359_1, (864, ), (1, ))
    assert_size_stride(arg1360_1, (), ())
    assert_size_stride(arg1361_1, (864, ), (1, ))
    assert_size_stride(arg1362_1, (864, ), (1, ))
    assert_size_stride(arg1363_1, (), ())
    assert_size_stride(arg1364_1, (864, ), (1, ))
    assert_size_stride(arg1365_1, (864, ), (1, ))
    assert_size_stride(arg1366_1, (), ())
    assert_size_stride(arg1367_1, (864, ), (1, ))
    assert_size_stride(arg1368_1, (864, ), (1, ))
    assert_size_stride(arg1369_1, (), ())
    assert_size_stride(arg1370_1, (864, ), (1, ))
    assert_size_stride(arg1371_1, (864, ), (1, ))
    assert_size_stride(arg1372_1, (), ())
    assert_size_stride(arg1373_1, (864, ), (1, ))
    assert_size_stride(arg1374_1, (864, ), (1, ))
    assert_size_stride(arg1375_1, (), ())
    assert_size_stride(arg1376_1, (864, ), (1, ))
    assert_size_stride(arg1377_1, (864, ), (1, ))
    assert_size_stride(arg1378_1, (), ())
    assert_size_stride(arg1379_1, (8, 3, 331, 331), (328683, 109561, 331, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 331, 331), (328683, 1, 993, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg1379_1, buf0, 24, 109561, grid=grid(24, 109561), stream=stream0)
        del arg1379_1
        buf1 = empty_strided((96, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg26_1, buf1, 288, 9, grid=grid(288, 9), stream=stream0)
        del arg26_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 96, 165, 165), (2613600, 27225, 165, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf5 = empty_strided((8, 96, 165, 165), (2613600, 1, 15840, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg777_1, arg778_1, arg0_1, arg1_1, buf5, 768, 27225, grid=grid(768, 27225), stream=stream0)
        del arg0_1
        del arg1_1
        del arg777_1
        del arg778_1
        buf4 = empty_strided((8, 96, 83, 83), (661344, 1, 7968, 96), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((8, 96, 83, 83), (661344, 1, 7968, 96), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((8, 96, 83, 83), (661344, 1, 7968, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool, l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool, l__mod___cell_stem_1_conv_prev_1x1_path_2_pad, max_pool2d, x_21, x_89], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3.run(buf3, buf4, buf10, buf12, 768, 6889, grid=grid(768, 6889), stream=stream0)
        # Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 54, 165, 165), (1470150, 27225, 165, 1))
        del arg27_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf67 = empty_strided((8, 54, 165, 165), (1470150, 1, 8910, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84, x_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf7, arg779_1, arg780_1, arg28_1, arg29_1, buf67, 432, 27225, grid=grid(432, 27225), stream=stream0)
        del arg28_1
        del arg29_1
        del arg779_1
        del arg780_1
        buf26 = empty_strided((8, 54, 171, 171), (1579014, 1, 9234, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22, x_24], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_5.run(buf7, buf26, 432, 29241, grid=grid(432, 29241), stream=stream0)
        # Source Nodes: [x_22, x_24, x_25], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf27 = extern_kernels.convolution(buf26, arg3_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf27, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg3_1
        del buf26
        buf28 = empty_strided((8, 54, 83, 83), (372006, 1, 4482, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf27, buf28, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf27
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg40_1
        buf30 = buf28; del buf28  # reuse
        # Source Nodes: [x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf29, arg791_1, arg792_1, arg41_1, arg42_1, buf30, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg41_1
        del arg42_1
        del arg791_1
        del arg792_1
        del buf29
        # Source Nodes: [x_28, x_29, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf31 = extern_kernels.convolution(buf30, arg43_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf31, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg43_1
        buf32 = buf30; del buf30  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf31, buf32, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf31
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg44_1
        buf34 = empty_strided((8, 54, 169, 169), (1542294, 1, 9126, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_38], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_8.run(buf7, buf34, 432, 28561, grid=grid(432, 28561), stream=stream0)
        # Source Nodes: [x_36, x_38, x_39], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg4_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf35, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg4_1
        del buf34
        buf36 = buf32; del buf32  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf35, buf36, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf35
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg47_1
        buf38 = buf36; del buf36  # reuse
        # Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf37, arg797_1, arg798_1, arg48_1, arg49_1, buf38, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg48_1
        del arg49_1
        del arg797_1
        del arg798_1
        del buf37
        # Source Nodes: [x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf39 = extern_kernels.convolution(buf38, arg50_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf39, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg50_1
        buf40 = buf38; del buf38  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf39, buf40, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf39
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg51_1
        buf42 = empty_strided((8, 54, 167, 167), (1506006, 1, 9018, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48, x_50], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_9.run(buf7, buf42, 432, 27889, grid=grid(432, 27889), stream=stream0)
        # Source Nodes: [x_48, x_50, x_51], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg5_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf43, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg5_1
        del buf42
        buf44 = buf40; del buf40  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf43, buf44, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf43
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg54_1
        buf46 = buf44; del buf44  # reuse
        # Source Nodes: [x_54, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf45, arg803_1, arg804_1, arg55_1, arg56_1, buf46, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg55_1
        del arg56_1
        del arg803_1
        del arg804_1
        del buf45
        # Source Nodes: [x_54, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf46, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf47, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg57_1
        buf48 = buf46; del buf46  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf47, buf48, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf47
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg58_1
        buf72 = empty((8, 270, 83, 83), device='cuda', dtype=torch.float32)
        buf50 = reinterpret_tensor(buf72, (8, 54, 83, 83), (1860030, 6889, 83, 1), 744012)  # alias
        buf51 = buf48; del buf48  # reuse
        # Source Nodes: [x_60, x_comb_iter_2, x_comb_iter_2_left, x_comb_iter_2_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf41, arg800_1, arg801_1, arg52_1, arg53_1, buf49, arg806_1, arg807_1, arg59_1, arg60_1, buf50, buf51, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg52_1
        del arg53_1
        del arg59_1
        del arg60_1
        del arg800_1
        del arg801_1
        del arg806_1
        del arg807_1
        del buf41
        del buf49
        # Source Nodes: [x_60, x_61], Original ATen: [aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf52, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg61_1
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf52, buf53, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf52
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg62_1
        buf55 = buf53; del buf53  # reuse
        # Source Nodes: [x_64, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf54, arg809_1, arg810_1, arg63_1, arg64_1, buf55, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg63_1
        del arg64_1
        del arg809_1
        del arg810_1
        del buf54
        # Source Nodes: [x_64, x_65, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf56, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg65_1
        buf57 = buf55; del buf55  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf56, buf57, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf56
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg66_1
        del buf57
        buf70 = reinterpret_tensor(buf72, (8, 54, 83, 83), (1860030, 6889, 83, 1), 372006)  # alias
        buf71 = reinterpret_tensor(buf72, (8, 54, 83, 83), (1860030, 6889, 83, 1), 1116018)  # alias
        # Source Nodes: [x_35, x_71, x_comb_iter_1, x_comb_iter_1_left, x_comb_iter_1_right, x_comb_iter_3, x_comb_iter_3_left, x_comb_iter_3_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_11.run(buf7, buf33, arg794_1, arg795_1, arg45_1, arg46_1, buf58, arg812_1, arg813_1, arg67_1, arg68_1, buf70, buf71, 2976048, grid=grid(2976048), stream=stream0)
        del arg45_1
        del arg46_1
        del arg67_1
        del arg68_1
        del arg794_1
        del arg795_1
        del arg812_1
        del arg813_1
        del buf33
        del buf58
        del buf7
        # Source Nodes: [l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool, x_89, x_path1], Original ATen: [aten.avg_pool2d, aten.convolution, aten.relu]
        buf11 = extern_kernels.convolution(buf10, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg78_1
        del buf10
        # Source Nodes: [l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool, l__mod___cell_stem_1_conv_prev_1x1_path_2_pad, x_89, x_path2], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf12, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg79_1
        buf14 = empty((8, 108, 83, 83), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_34, x_left], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_12.run(buf11, buf13, arg824_1, arg825_1, arg80_1, arg81_1, buf14, 5952096, grid=grid(5952096), stream=stream0)
        del arg80_1
        del arg81_1
        del arg824_1
        del arg825_1
        del buf11
        buf84 = empty_strided((8, 108, 87, 87), (817452, 1, 9396, 108), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_95], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_13.run(buf14, buf84, 864, 7569, grid=grid(864, 7569), stream=stream0)
        # Source Nodes: [x_93, x_95, x_96], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf85 = extern_kernels.convolution(buf84, arg8_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf85, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg8_1
        buf86 = empty_strided((8, 108, 42, 42), (190512, 1, 4536, 108), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf85, buf86, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf85
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg85_1
        buf88 = buf86; del buf86  # reuse
        # Source Nodes: [x_100, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf87, arg830_1, arg831_1, arg86_1, arg87_1, buf88, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg830_1
        del arg831_1
        del arg86_1
        del arg87_1
        del buf87
        # Source Nodes: [x_100, x_101, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf89 = extern_kernels.convolution(buf88, arg88_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf89, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg88_1
        buf90 = buf88; del buf88  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf89, buf90, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf89
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg89_1
        del buf90
        buf139 = empty((8, 540, 42, 42), device='cuda', dtype=torch.float32)
        buf136 = reinterpret_tensor(buf139, (8, 108, 42, 42), (952560, 1764, 42, 1), 0)  # alias
        # Source Nodes: [x_106, x_comb_iter_0_left_1, x_comb_iter_0_right_1, x_comb_iter_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_16.run(buf14, buf91, arg833_1, arg834_1, arg90_1, arg91_1, buf136, 1524096, grid=grid(1524096), stream=stream0)
        del arg833_1
        del arg834_1
        del arg90_1
        del arg91_1
        buf16 = empty_strided((8, 96, 169, 169), (2741856, 1, 16224, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_8], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_17.run(buf3, buf16, 768, 28561, grid=grid(768, 28561), stream=stream0)
        # Source Nodes: [x_10, x_11, x_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf17 = extern_kernels.convolution(buf16, arg2_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf17, (8, 96, 83, 83), (661344, 6889, 83, 1))
        del arg2_1
        del buf16
        buf18 = buf12; del buf12  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf17, buf18, 768, 6889, grid=grid(768, 6889), stream=stream0)
        del buf17
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg30_1
        del buf18
        buf20 = reinterpret_tensor(buf13, (8, 54, 83, 83), (372006, 1, 4482, 54), 0); del buf13  # reuse
        # Source Nodes: [x_14, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf19, arg782_1, arg783_1, arg31_1, arg32_1, buf20, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg31_1
        del arg32_1
        del arg782_1
        del arg783_1
        del buf19
        # Source Nodes: [x_14, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf21 = extern_kernels.convolution(buf20, arg33_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf21, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg33_1
        buf22 = buf20; del buf20  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf21, buf22, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf21
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg34_1
        del buf22
        # Source Nodes: [l__mod___cell_stem_0_comb_iter_0_right_conv], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf4, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg37_1
        buf25 = reinterpret_tensor(buf72, (8, 54, 83, 83), (1860030, 6889, 83, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0, x_comb_iter_0_left, x_comb_iter_0_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf23, arg785_1, arg786_1, arg35_1, arg36_1, buf24, arg788_1, arg789_1, arg38_1, arg39_1, buf25, 2976048, grid=grid(2976048), stream=stream0)
        del arg35_1
        del arg36_1
        del arg38_1
        del arg39_1
        del arg785_1
        del arg786_1
        del arg788_1
        del arg789_1
        del buf23
        buf59 = empty_strided((8, 96, 167, 167), (2677344, 1, 16032, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72, x_74], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_20.run(buf3, buf59, 768, 27889, grid=grid(768, 27889), stream=stream0)
        del buf3
        # Source Nodes: [x_72, x_74, x_75], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg6_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf60, (8, 96, 83, 83), (661344, 6889, 83, 1))
        del arg6_1
        del buf59
        buf61 = buf4; del buf4  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf60, buf61, 768, 6889, grid=grid(768, 6889), stream=stream0)
        del buf60
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg69_1
        del buf61
        buf63 = reinterpret_tensor(buf24, (8, 54, 83, 83), (372006, 1, 4482, 54), 0); del buf24  # reuse
        # Source Nodes: [x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf62, arg815_1, arg816_1, arg70_1, arg71_1, buf63, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del arg70_1
        del arg71_1
        del arg815_1
        del arg816_1
        del buf62
        # Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf64, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg72_1
        buf65 = buf63; del buf63  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf64, buf65, 432, 6889, grid=grid(432, 6889), stream=stream0)
        del buf64
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg73_1
        del buf65
        # Source Nodes: [x_84, x_87], Original ATen: [aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf67, arg7_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 54, 83, 83), (372006, 6889, 83, 1))
        del arg7_1
        del buf67
        buf69 = reinterpret_tensor(buf72, (8, 54, 83, 83), (1860030, 6889, 83, 1), 1488024)  # alias
        # Source Nodes: [x_comb_iter_4, x_comb_iter_4_left, x_comb_iter_4_right], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf66, arg818_1, arg819_1, arg74_1, arg75_1, buf68, arg821_1, arg822_1, arg76_1, arg77_1, buf69, 2976048, grid=grid(2976048), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        del arg77_1
        del arg818_1
        del arg819_1
        del arg821_1
        del arg822_1
        del buf66
        del buf68
        buf73 = empty_strided((8, 270, 83, 83), (1860030, 1, 22410, 270), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten.relu]
        triton_poi_fused_relu_22.run(buf72, buf73, 2160, 6889, grid=grid(2160, 6889), stream=stream0)
        del buf25
        del buf50
        del buf69
        del buf70
        del buf71
        # Source Nodes: [x_90, x_91], Original ATen: [aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 108, 83, 83), (744012, 6889, 83, 1))
        del arg82_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        buf133 = empty_strided((8, 108, 83, 83), (744012, 1, 8964, 108), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169, x_right_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf75, arg827_1, arg828_1, arg83_1, arg84_1, buf133, 864, 6889, grid=grid(864, 6889), stream=stream0)
        del arg827_1
        del arg828_1
        del arg83_1
        del arg84_1
        buf100 = buf84; del buf84  # reuse
        # Source Nodes: [x_121, x_123], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_13.run(buf75, buf100, 864, 7569, grid=grid(864, 7569), stream=stream0)
        # Source Nodes: [x_121, x_123, x_124], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf101 = extern_kernels.convolution(buf100, arg10_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf101, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg10_1
        del buf100
        buf102 = reinterpret_tensor(buf91, (8, 108, 42, 42), (190512, 1, 4536, 108), 0); del buf91  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf101, buf102, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf101
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg99_1
        buf104 = buf102; del buf102  # reuse
        # Source Nodes: [x_127, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf103, arg842_1, arg843_1, arg100_1, arg101_1, buf104, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg100_1
        del arg101_1
        del arg842_1
        del arg843_1
        del buf103
        # Source Nodes: [x_127, x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf105 = extern_kernels.convolution(buf104, arg102_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf105, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg102_1
        buf106 = buf104; del buf104  # reuse
        # Source Nodes: [x_131], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf105, buf106, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf105
        # Source Nodes: [x_131], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg103_1
        buf108 = empty_strided((8, 108, 85, 85), (780300, 1, 9180, 108), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_135], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_24.run(buf75, buf108, 864, 7225, grid=grid(864, 7225), stream=stream0)
        # Source Nodes: [x_133, x_135, x_136], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf109 = extern_kernels.convolution(buf108, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf109, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg11_1
        buf110 = buf106; del buf106  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf109, buf110, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf109
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg106_1
        buf112 = buf110; del buf110  # reuse
        # Source Nodes: [x_139, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf111, arg848_1, arg849_1, arg107_1, arg108_1, buf112, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg107_1
        del arg108_1
        del arg848_1
        del arg849_1
        del buf111
        # Source Nodes: [x_139, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg109_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf113, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg109_1
        buf114 = buf112; del buf112  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf113, buf114, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf113
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg110_1
        buf116 = reinterpret_tensor(buf139, (8, 108, 42, 42), (952560, 1764, 42, 1), 381024)  # alias
        buf117 = buf114; del buf114  # reuse
        # Source Nodes: [x_145, x_comb_iter_2_left_1, x_comb_iter_2_right_1, x_comb_iter_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf107, arg845_1, arg846_1, arg104_1, arg105_1, buf115, arg851_1, arg852_1, arg111_1, arg112_1, buf116, buf117, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg104_1
        del arg105_1
        del arg111_1
        del arg112_1
        del arg845_1
        del arg846_1
        del arg851_1
        del arg852_1
        del buf107
        del buf115
        # Source Nodes: [x_145, x_146], Original ATen: [aten.convolution, aten.relu]
        buf118 = extern_kernels.convolution(buf117, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf118, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg113_1
        buf119 = buf117; del buf117  # reuse
        # Source Nodes: [x_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf118, buf119, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf118
        # Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg114_1
        buf121 = buf119; del buf119  # reuse
        # Source Nodes: [x_149, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf120, arg854_1, arg855_1, arg115_1, arg116_1, buf121, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg115_1
        del arg116_1
        del arg854_1
        del arg855_1
        del buf120
        # Source Nodes: [x_149, x_150, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf122 = extern_kernels.convolution(buf121, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf122, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg117_1
        buf123 = buf121; del buf121  # reuse
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf122, buf123, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf122
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg118_1
        buf92 = empty_strided((8, 108, 89, 89), (855468, 1, 9612, 108), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107, x_109], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_26.run(buf75, buf92, 864, 7921, grid=grid(864, 7921), stream=stream0)
        # Source Nodes: [x_107, x_109, x_110], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg9_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf93, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg9_1
        del buf92
        buf94 = buf123; del buf123  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf93, buf94, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf93
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg92_1
        buf96 = buf94; del buf94  # reuse
        # Source Nodes: [x_113, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf95, arg836_1, arg837_1, arg93_1, arg94_1, buf96, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg836_1
        del arg837_1
        del arg93_1
        del arg94_1
        del buf95
        # Source Nodes: [x_113, x_114, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf97 = extern_kernels.convolution(buf96, arg95_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf97, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg95_1
        buf98 = buf96; del buf96  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf97, buf98, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf97
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg96_1
        del buf98
        buf137 = reinterpret_tensor(buf139, (8, 108, 42, 42), (952560, 1764, 42, 1), 190512)  # alias
        buf138 = reinterpret_tensor(buf139, (8, 108, 42, 42), (952560, 1764, 42, 1), 571536)  # alias
        # Source Nodes: [x_120, x_156, x_comb_iter_1_left_1, x_comb_iter_1_right_1, x_comb_iter_3_left_1, x_comb_iter_3_right_1, x_comb_iter_6, x_comb_iter_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_27.run(buf75, buf99, arg839_1, arg840_1, arg97_1, arg98_1, buf124, arg857_1, arg858_1, arg119_1, arg120_1, buf137, buf138, 1524096, grid=grid(1524096), stream=stream0)
        del arg119_1
        del arg120_1
        del arg839_1
        del arg840_1
        del arg857_1
        del arg858_1
        del arg97_1
        del arg98_1
        del buf124
        del buf75
        del buf99
        buf78 = empty_strided((8, 270, 42, 42), (476280, 1, 11340, 270), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_1_avgpool, x_174], Original ATen: [aten.avg_pool2d, aten.relu]
        triton_poi_fused_avg_pool2d_relu_28.run(buf72, buf78, 2160, 1764, grid=grid(2160, 1764), stream=stream0)
        # Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_1_avgpool, x_174, x_path1_1], Original ATen: [aten.avg_pool2d, aten.convolution, aten.relu]
        buf79 = extern_kernels.convolution(buf78, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg130_1
        buf80 = buf78; del buf78  # reuse
        # Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_2_avgpool, l__mod___cell_0_conv_prev_1x1_path_2_pad, x_174], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_29.run(buf72, buf80, 2160, 1764, grid=grid(2160, 1764), stream=stream0)
        del buf72
        # Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_2_avgpool, l__mod___cell_0_conv_prev_1x1_path_2_pad, x_174, x_path2_1], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.relu]
        buf81 = extern_kernels.convolution(buf80, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg131_1
        buf82 = empty((8, 216, 42, 42), device='cuda', dtype=torch.float32)
        buf149 = empty_strided((8, 216, 42, 42), (381024, 1, 9072, 216), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((8, 216, 42, 42), (381024, 1, 9072, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_32, x_178, x_228, x_left_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30.run(buf79, buf81, arg869_1, arg870_1, arg132_1, arg133_1, buf82, buf149, buf190, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg132_1
        del arg133_1
        del arg869_1
        del arg870_1
        del buf79
        # Source Nodes: [x_178, x_179], Original ATen: [aten.convolution, aten.relu]
        buf150 = extern_kernels.convolution(buf149, arg137_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf150, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg137_1
        buf151 = buf149; del buf149  # reuse
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf150, buf151, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf150
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg138_1
        buf153 = buf151; del buf151  # reuse
        # Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf152, arg875_1, arg876_1, arg139_1, arg140_1, buf153, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg139_1
        del arg140_1
        del arg875_1
        del arg876_1
        del buf152
        # Source Nodes: [x_182, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf154 = extern_kernels.convolution(buf153, arg141_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf154, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg141_1
        buf155 = buf153; del buf153  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf154, buf155, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf154
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg142_1
        del buf155
        buf202 = empty((8, 1080, 42, 42), device='cuda', dtype=torch.float32)
        buf198 = reinterpret_tensor(buf202, (8, 216, 42, 42), (1905120, 1764, 42, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_2, x_comb_iter_0_right_2, x_comb_iter_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33.run(buf82, buf156, arg878_1, arg879_1, arg143_1, arg144_1, buf198, 3048192, grid=grid(3048192), stream=stream0)
        del arg143_1
        del arg144_1
        del arg878_1
        del arg879_1
        del buf156
        buf125 = buf108; del buf108  # reuse
        # Source Nodes: [x_157, x_159], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_24.run(buf14, buf125, 864, 7225, grid=grid(864, 7225), stream=stream0)
        del buf14
        # Source Nodes: [x_157, x_159, x_160], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg12_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf126, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg12_1
        del buf125
        buf127 = reinterpret_tensor(buf81, (8, 108, 42, 42), (190512, 1, 4536, 108), 0); del buf81  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf126, buf127, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf126
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg121_1
        buf129 = buf127; del buf127  # reuse
        # Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf128, arg860_1, arg861_1, arg122_1, arg123_1, buf129, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del arg122_1
        del arg123_1
        del arg860_1
        del arg861_1
        del buf128
        # Source Nodes: [x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf130 = extern_kernels.convolution(buf129, arg124_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf130, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg124_1
        buf131 = buf129; del buf129  # reuse
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf130, buf131, 864, 1764, grid=grid(864, 1764), stream=stream0)
        del buf130
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg125_1
        del buf131
        # Source Nodes: [x_169, x_172], Original ATen: [aten.convolution, aten.relu]
        buf134 = extern_kernels.convolution(buf133, arg13_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 108, 42, 42), (190512, 1764, 42, 1))
        del arg13_1
        del buf133
        buf135 = reinterpret_tensor(buf139, (8, 108, 42, 42), (952560, 1764, 42, 1), 762048)  # alias
        # Source Nodes: [x_comb_iter_4_left_1, x_comb_iter_4_right_1, x_comb_iter_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf132, arg863_1, arg864_1, arg126_1, arg127_1, buf134, arg866_1, arg867_1, arg128_1, arg129_1, buf135, 1524096, grid=grid(1524096), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del arg863_1
        del arg864_1
        del arg866_1
        del arg867_1
        del buf132
        buf140 = empty_strided((8, 540, 42, 42), (952560, 1, 22680, 540), device='cuda', dtype=torch.float32)
        buf145 = empty_strided((8, 540, 42, 42), (952560, 1, 22680, 540), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_238], Original ATen: [aten.relu]
        triton_poi_fused_relu_35.run(buf139, buf140, buf145, 4320, 1764, grid=grid(4320, 1764), stream=stream0)
        del buf116
        del buf135
        del buf136
        del buf137
        del buf138
        # Source Nodes: [x_175, x_176], Original ATen: [aten.convolution, aten.relu]
        buf141 = extern_kernels.convolution(buf140, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg134_1
        buf142 = buf141; del buf141  # reuse
        # Source Nodes: [x_comb_iter_4_right_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf142, arg872_1, arg873_1, arg135_1, arg136_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg135_1
        del arg136_1
        del arg872_1
        del arg873_1
        # Source Nodes: [x_228, x_229], Original ATen: [aten.convolution, aten.relu]
        buf191 = extern_kernels.convolution(buf190, arg177_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf191, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg177_1
        buf192 = buf190; del buf190  # reuse
        # Source Nodes: [x_231], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf191, buf192, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_231], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg178_1
        buf194 = buf192; del buf192  # reuse
        # Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf193, arg905_1, arg906_1, arg179_1, arg180_1, buf194, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg179_1
        del arg180_1
        del arg905_1
        del arg906_1
        # Source Nodes: [x_232, x_233, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf195 = extern_kernels.convolution(buf194, arg181_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf195, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg181_1
        buf196 = buf194; del buf194  # reuse
        # Source Nodes: [x_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf195, buf196, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_236], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg182_1
        buf143 = reinterpret_tensor(buf196, (8, 216, 42, 42), (381024, 1764, 42, 1), 0); del buf196  # reuse
        buf144 = buf195; del buf195  # reuse
        buf157 = reinterpret_tensor(buf193, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf193  # reuse
        buf165 = reinterpret_tensor(buf191, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf191  # reuse
        buf173 = reinterpret_tensor(buf82, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf82  # reuse
        buf201 = reinterpret_tensor(buf202, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1524096)  # alias
        # Source Nodes: [x_188, x_198, x_208, x_comb_iter_14, x_comb_iter_1_right_2, x_comb_iter_3_right_2, x_comb_iter_4_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37.run(buf142, buf197, arg908_1, arg909_1, arg183_1, arg184_1, buf143, buf144, buf157, buf165, buf173, buf201, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg183_1
        del arg184_1
        del arg908_1
        del arg909_1
        # Source Nodes: [x_238, x_239], Original ATen: [aten.convolution, aten.relu]
        buf146 = extern_kernels.convolution(buf145, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg185_1
        buf147 = buf146; del buf146  # reuse
        buf212 = reinterpret_tensor(buf197, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf197  # reuse
        buf253 = reinterpret_tensor(buf142, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf142  # reuse
        # Source Nodes: [x_244, x_294, x_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf147, arg911_1, arg912_1, arg186_1, arg187_1, buf212, buf253, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg186_1
        del arg187_1
        del arg911_1
        del arg912_1
        # Source Nodes: [x_244, x_245], Original ATen: [aten.convolution, aten.relu]
        buf213 = extern_kernels.convolution(buf212, arg191_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf213, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg191_1
        buf214 = buf212; del buf212  # reuse
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf213, buf214, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf213
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg192_1
        buf216 = buf214; del buf214  # reuse
        # Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf215, arg917_1, arg918_1, arg193_1, arg194_1, buf216, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg193_1
        del arg194_1
        del arg917_1
        del arg918_1
        del buf215
        # Source Nodes: [x_248, x_249, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf217 = extern_kernels.convolution(buf216, arg195_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf217, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg195_1
        buf218 = buf216; del buf216  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf217, buf218, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf217
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg196_1
        del buf218
        buf265 = empty((8, 1080, 42, 42), device='cuda', dtype=torch.float32)
        buf261 = reinterpret_tensor(buf265, (8, 216, 42, 42), (1905120, 1764, 42, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_3, x_comb_iter_0_right_3, x_comb_iter_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33.run(buf147, buf219, arg920_1, arg921_1, arg197_1, arg198_1, buf261, 3048192, grid=grid(3048192), stream=stream0)
        del arg197_1
        del arg198_1
        del arg920_1
        del arg921_1
        del buf147
        del buf219
        # Source Nodes: [x_188, x_189], Original ATen: [aten.convolution, aten.relu]
        buf158 = extern_kernels.convolution(buf157, arg145_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf158, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg145_1
        buf159 = buf157; del buf157  # reuse
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf158, buf159, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf158
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg146_1
        buf161 = buf159; del buf159  # reuse
        # Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf160, arg881_1, arg882_1, arg147_1, arg148_1, buf161, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg147_1
        del arg148_1
        del arg881_1
        del arg882_1
        del buf160
        # Source Nodes: [x_192, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf162 = extern_kernels.convolution(buf161, arg149_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf162, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg149_1
        buf163 = buf161; del buf161  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf162, buf163, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf162
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg150_1
        del buf163
        # Source Nodes: [x_198, x_199], Original ATen: [aten.convolution, aten.relu]
        buf166 = extern_kernels.convolution(buf165, arg153_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf166, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg153_1
        buf167 = buf165; del buf165  # reuse
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf166, buf167, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf166
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg154_1
        buf169 = buf167; del buf167  # reuse
        # Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf168, arg887_1, arg888_1, arg155_1, arg156_1, buf169, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg155_1
        del arg156_1
        del arg887_1
        del arg888_1
        del buf168
        # Source Nodes: [x_202, x_203, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf170 = extern_kernels.convolution(buf169, arg157_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf170, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg157_1
        buf171 = buf169; del buf169  # reuse
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf170, buf171, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf170
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg158_1
        del buf171
        # Source Nodes: [x_208, x_209], Original ATen: [aten.convolution, aten.relu]
        buf174 = extern_kernels.convolution(buf173, arg161_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf174, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg161_1
        buf175 = buf173; del buf173  # reuse
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf174, buf175, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf174
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg162_1
        buf177 = buf175; del buf175  # reuse
        # Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf176, arg893_1, arg894_1, arg163_1, arg164_1, buf177, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg163_1
        del arg164_1
        del arg893_1
        del arg894_1
        del buf176
        # Source Nodes: [x_212, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf178 = extern_kernels.convolution(buf177, arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf178, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg165_1
        buf179 = buf177; del buf177  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf178, buf179, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf178
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg166_1
        buf181 = reinterpret_tensor(buf202, (8, 216, 42, 42), (1905120, 1764, 42, 1), 762048)  # alias
        buf182 = buf179; del buf179  # reuse
        # Source Nodes: [x_218, x_comb_iter_12, x_comb_iter_2_left_2, x_comb_iter_2_right_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf172, arg890_1, arg891_1, arg159_1, arg160_1, buf180, arg896_1, arg897_1, arg167_1, arg168_1, buf181, buf182, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg159_1
        del arg160_1
        del arg167_1
        del arg168_1
        del arg890_1
        del arg891_1
        del arg896_1
        del arg897_1
        del buf172
        del buf180
        # Source Nodes: [x_218, x_219], Original ATen: [aten.convolution, aten.relu]
        buf183 = extern_kernels.convolution(buf182, arg169_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf183, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg169_1
        buf184 = buf182; del buf182  # reuse
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf183, buf184, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf183
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg170_1
        buf186 = buf184; del buf184  # reuse
        # Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf185, arg899_1, arg900_1, arg171_1, arg172_1, buf186, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg171_1
        del arg172_1
        del arg899_1
        del arg900_1
        del buf185
        # Source Nodes: [x_222, x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf187 = extern_kernels.convolution(buf186, arg173_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf187, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg173_1
        buf188 = buf186; del buf186  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf187, buf188, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf187
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg174_1
        del buf188
        buf199 = reinterpret_tensor(buf202, (8, 216, 42, 42), (1905120, 1764, 42, 1), 381024)  # alias
        # Source Nodes: [x_comb_iter_11, x_comb_iter_1_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf164, arg884_1, arg885_1, arg151_1, arg152_1, buf143, buf199, 3048192, grid=grid(3048192), stream=stream0)
        del arg151_1
        del arg152_1
        del arg884_1
        del arg885_1
        del buf143
        del buf164
        buf200 = reinterpret_tensor(buf202, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1143072)  # alias
        # Source Nodes: [x_comb_iter_13, x_comb_iter_3_left_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf189, arg902_1, arg903_1, arg175_1, arg176_1, buf144, buf200, 3048192, grid=grid(3048192), stream=stream0)
        del arg175_1
        del arg176_1
        del arg902_1
        del arg903_1
        del buf144
        buf203 = empty_strided((8, 1080, 42, 42), (1905120, 1, 45360, 1080), device='cuda', dtype=torch.float32)
        buf208 = empty_strided((8, 1080, 42, 42), (1905120, 1, 45360, 1080), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241, x_304], Original ATen: [aten.relu]
        triton_poi_fused_relu_41.run(buf202, buf203, buf208, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del buf181
        del buf198
        del buf199
        del buf200
        del buf201
        # Source Nodes: [x_241, x_242], Original ATen: [aten.convolution, aten.relu]
        buf204 = extern_kernels.convolution(buf203, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg188_1
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [x_comb_iter_4_right_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf205, arg914_1, arg915_1, arg189_1, arg190_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg189_1
        del arg190_1
        del arg914_1
        del arg915_1
        # Source Nodes: [x_294, x_295], Original ATen: [aten.convolution, aten.relu]
        buf254 = extern_kernels.convolution(buf253, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf254, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg231_1
        buf255 = buf253; del buf253  # reuse
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf254, buf255, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg232_1
        buf257 = buf255; del buf255  # reuse
        # Source Nodes: [x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf256, arg947_1, arg948_1, arg233_1, arg234_1, buf257, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg233_1
        del arg234_1
        del arg947_1
        del arg948_1
        # Source Nodes: [x_298, x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf258 = extern_kernels.convolution(buf257, arg235_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf258, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg235_1
        buf259 = buf257; del buf257  # reuse
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf258, buf259, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg236_1
        buf206 = reinterpret_tensor(buf259, (8, 216, 42, 42), (381024, 1764, 42, 1), 0); del buf259  # reuse
        buf207 = buf258; del buf258  # reuse
        buf220 = reinterpret_tensor(buf256, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf256  # reuse
        buf228 = reinterpret_tensor(buf254, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf254  # reuse
        buf236 = reinterpret_tensor(buf189, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf189  # reuse
        buf264 = reinterpret_tensor(buf265, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1524096)  # alias
        # Source Nodes: [x_254, x_264, x_274, x_comb_iter_19, x_comb_iter_1_right_3, x_comb_iter_3_right_3, x_comb_iter_4_left_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37.run(buf205, buf260, arg950_1, arg951_1, arg237_1, arg238_1, buf206, buf207, buf220, buf228, buf236, buf264, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg237_1
        del arg238_1
        del arg950_1
        del arg951_1
        # Source Nodes: [x_304, x_305], Original ATen: [aten.convolution, aten.relu]
        buf209 = extern_kernels.convolution(buf208, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg239_1
        buf210 = buf209; del buf209  # reuse
        buf275 = reinterpret_tensor(buf260, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf260  # reuse
        buf316 = reinterpret_tensor(buf205, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf205  # reuse
        # Source Nodes: [x_310, x_360, x_left_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf210, arg953_1, arg954_1, arg240_1, arg241_1, buf275, buf316, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg240_1
        del arg241_1
        del arg953_1
        del arg954_1
        # Source Nodes: [x_310, x_311], Original ATen: [aten.convolution, aten.relu]
        buf276 = extern_kernels.convolution(buf275, arg245_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf276, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg245_1
        buf277 = buf275; del buf275  # reuse
        # Source Nodes: [x_313], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf276, buf277, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf276
        # Source Nodes: [x_313], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg246_1
        buf279 = buf277; del buf277  # reuse
        # Source Nodes: [x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf278, arg959_1, arg960_1, arg247_1, arg248_1, buf279, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg247_1
        del arg248_1
        del arg959_1
        del arg960_1
        del buf278
        # Source Nodes: [x_314, x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf280 = extern_kernels.convolution(buf279, arg249_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf280, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg249_1
        buf281 = buf279; del buf279  # reuse
        # Source Nodes: [x_318], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf280, buf281, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf280
        # Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg250_1
        del buf281
        buf328 = reinterpret_tensor(buf208, (8, 1080, 42, 42), (1905120, 1764, 42, 1), 0); del buf208  # reuse
        buf324 = reinterpret_tensor(buf328, (8, 216, 42, 42), (1905120, 1764, 42, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_4, x_comb_iter_0_right_4, x_comb_iter_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33.run(buf210, buf282, arg962_1, arg963_1, arg251_1, arg252_1, buf324, 3048192, grid=grid(3048192), stream=stream0)
        del arg251_1
        del arg252_1
        del arg962_1
        del arg963_1
        del buf210
        del buf282
        # Source Nodes: [x_254, x_255], Original ATen: [aten.convolution, aten.relu]
        buf221 = extern_kernels.convolution(buf220, arg199_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf221, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg199_1
        buf222 = buf220; del buf220  # reuse
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf221, buf222, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf221
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg200_1
        buf224 = buf222; del buf222  # reuse
        # Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf223, arg923_1, arg924_1, arg201_1, arg202_1, buf224, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg201_1
        del arg202_1
        del arg923_1
        del arg924_1
        del buf223
        # Source Nodes: [x_258, x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf225 = extern_kernels.convolution(buf224, arg203_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf225, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg203_1
        buf226 = buf224; del buf224  # reuse
        # Source Nodes: [x_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf225, buf226, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf225
        # Source Nodes: [x_262], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg204_1
        del buf226
        # Source Nodes: [x_264, x_265], Original ATen: [aten.convolution, aten.relu]
        buf229 = extern_kernels.convolution(buf228, arg207_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf229, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg207_1
        buf230 = buf228; del buf228  # reuse
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf229, buf230, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf229
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg208_1
        buf232 = buf230; del buf230  # reuse
        # Source Nodes: [x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf231, arg929_1, arg930_1, arg209_1, arg210_1, buf232, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg209_1
        del arg210_1
        del arg929_1
        del arg930_1
        del buf231
        # Source Nodes: [x_268, x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf233 = extern_kernels.convolution(buf232, arg211_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf233, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg211_1
        buf234 = buf232; del buf232  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf233, buf234, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf233
        # Source Nodes: [x_272], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg212_1
        del buf234
        # Source Nodes: [x_274, x_275], Original ATen: [aten.convolution, aten.relu]
        buf237 = extern_kernels.convolution(buf236, arg215_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf237, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg215_1
        buf238 = buf236; del buf236  # reuse
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf237, buf238, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf237
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg216_1
        buf240 = buf238; del buf238  # reuse
        # Source Nodes: [x_278, x_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf239, arg935_1, arg936_1, arg217_1, arg218_1, buf240, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg217_1
        del arg218_1
        del arg935_1
        del arg936_1
        del buf239
        # Source Nodes: [x_278, x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf241 = extern_kernels.convolution(buf240, arg219_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf241, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg219_1
        buf242 = buf240; del buf240  # reuse
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf241, buf242, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf241
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg220_1
        buf244 = reinterpret_tensor(buf265, (8, 216, 42, 42), (1905120, 1764, 42, 1), 762048)  # alias
        buf245 = buf242; del buf242  # reuse
        # Source Nodes: [x_284, x_comb_iter_17, x_comb_iter_2_left_3, x_comb_iter_2_right_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf235, arg932_1, arg933_1, arg213_1, arg214_1, buf243, arg938_1, arg939_1, arg221_1, arg222_1, buf244, buf245, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg213_1
        del arg214_1
        del arg221_1
        del arg222_1
        del arg932_1
        del arg933_1
        del arg938_1
        del arg939_1
        del buf235
        del buf243
        # Source Nodes: [x_284, x_285], Original ATen: [aten.convolution, aten.relu]
        buf246 = extern_kernels.convolution(buf245, arg223_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf246, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg223_1
        buf247 = buf245; del buf245  # reuse
        # Source Nodes: [x_287], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf246, buf247, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf246
        # Source Nodes: [x_287], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg224_1
        buf249 = buf247; del buf247  # reuse
        # Source Nodes: [x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf248, arg941_1, arg942_1, arg225_1, arg226_1, buf249, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg225_1
        del arg226_1
        del arg941_1
        del arg942_1
        del buf248
        # Source Nodes: [x_288, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf250 = extern_kernels.convolution(buf249, arg227_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf250, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg227_1
        buf251 = buf249; del buf249  # reuse
        # Source Nodes: [x_292], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf250, buf251, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf250
        # Source Nodes: [x_292], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg228_1
        del buf251
        buf262 = reinterpret_tensor(buf265, (8, 216, 42, 42), (1905120, 1764, 42, 1), 381024)  # alias
        # Source Nodes: [x_comb_iter_16, x_comb_iter_1_left_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf227, arg926_1, arg927_1, arg205_1, arg206_1, buf206, buf262, 3048192, grid=grid(3048192), stream=stream0)
        del arg205_1
        del arg206_1
        del arg926_1
        del arg927_1
        del buf206
        del buf227
        buf263 = reinterpret_tensor(buf265, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1143072)  # alias
        # Source Nodes: [x_comb_iter_18, x_comb_iter_3_left_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf252, arg944_1, arg945_1, arg229_1, arg230_1, buf207, buf263, 3048192, grid=grid(3048192), stream=stream0)
        del arg229_1
        del arg230_1
        del arg944_1
        del arg945_1
        del buf207
        buf266 = buf203; del buf203  # reuse
        buf271 = reinterpret_tensor(buf202, (8, 1080, 42, 42), (1905120, 1, 45360, 1080), 0); del buf202  # reuse
        # Source Nodes: [x_307, x_370], Original ATen: [aten.relu]
        triton_poi_fused_relu_41.run(buf265, buf266, buf271, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del buf244
        del buf261
        del buf262
        del buf263
        del buf264
        # Source Nodes: [x_307, x_308], Original ATen: [aten.convolution, aten.relu]
        buf267 = extern_kernels.convolution(buf266, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg242_1
        buf268 = buf267; del buf267  # reuse
        # Source Nodes: [x_comb_iter_4_right_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf268, arg956_1, arg957_1, arg243_1, arg244_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg243_1
        del arg244_1
        del arg956_1
        del arg957_1
        # Source Nodes: [x_360, x_361], Original ATen: [aten.convolution, aten.relu]
        buf317 = extern_kernels.convolution(buf316, arg285_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf317, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg285_1
        buf318 = buf316; del buf316  # reuse
        # Source Nodes: [x_363], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf317, buf318, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_363], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg286_1
        buf320 = buf318; del buf318  # reuse
        # Source Nodes: [x_364, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf319, arg989_1, arg990_1, arg287_1, arg288_1, buf320, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg287_1
        del arg288_1
        del arg989_1
        del arg990_1
        # Source Nodes: [x_364, x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf321 = extern_kernels.convolution(buf320, arg289_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf321, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg289_1
        buf322 = buf320; del buf320  # reuse
        # Source Nodes: [x_368], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf321, buf322, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_368], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg290_1
        buf269 = reinterpret_tensor(buf322, (8, 216, 42, 42), (381024, 1764, 42, 1), 0); del buf322  # reuse
        buf270 = buf321; del buf321  # reuse
        buf283 = reinterpret_tensor(buf319, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf319  # reuse
        buf291 = reinterpret_tensor(buf317, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf317  # reuse
        buf299 = reinterpret_tensor(buf252, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf252  # reuse
        buf327 = reinterpret_tensor(buf328, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1524096)  # alias
        # Source Nodes: [x_320, x_330, x_340, x_comb_iter_1_right_4, x_comb_iter_24, x_comb_iter_3_right_4, x_comb_iter_4_left_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37.run(buf268, buf323, arg992_1, arg993_1, arg291_1, arg292_1, buf269, buf270, buf283, buf291, buf299, buf327, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg291_1
        del arg292_1
        del arg992_1
        del arg993_1
        # Source Nodes: [x_370, x_371], Original ATen: [aten.convolution, aten.relu]
        buf272 = extern_kernels.convolution(buf271, arg293_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg293_1
        buf273 = buf272; del buf272  # reuse
        buf338 = reinterpret_tensor(buf323, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf323  # reuse
        buf379 = reinterpret_tensor(buf268, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf268  # reuse
        # Source Nodes: [x_376, x_426, x_left_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf273, arg995_1, arg996_1, arg294_1, arg295_1, buf338, buf379, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg294_1
        del arg295_1
        del arg995_1
        del arg996_1
        # Source Nodes: [x_376, x_377], Original ATen: [aten.convolution, aten.relu]
        buf339 = extern_kernels.convolution(buf338, arg299_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf339, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg299_1
        buf340 = buf338; del buf338  # reuse
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf339, buf340, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf339
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg300_1
        buf342 = buf340; del buf340  # reuse
        # Source Nodes: [x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf341, arg1001_1, arg1002_1, arg301_1, arg302_1, buf342, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1001_1
        del arg1002_1
        del arg301_1
        del arg302_1
        del buf341
        # Source Nodes: [x_380, x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf343 = extern_kernels.convolution(buf342, arg303_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf343, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg303_1
        buf344 = buf342; del buf342  # reuse
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf343, buf344, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf343
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg304_1
        del buf344
        buf391 = reinterpret_tensor(buf271, (8, 1080, 42, 42), (1905120, 1764, 42, 1), 0); del buf271  # reuse
        buf387 = reinterpret_tensor(buf391, (8, 216, 42, 42), (1905120, 1764, 42, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_5, x_comb_iter_0_right_5, x_comb_iter_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_33.run(buf273, buf345, arg1004_1, arg1005_1, arg305_1, arg306_1, buf387, 3048192, grid=grid(3048192), stream=stream0)
        del arg1004_1
        del arg1005_1
        del arg305_1
        del arg306_1
        del buf273
        del buf345
        # Source Nodes: [x_320, x_321], Original ATen: [aten.convolution, aten.relu]
        buf284 = extern_kernels.convolution(buf283, arg253_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf284, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg253_1
        buf285 = buf283; del buf283  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf284, buf285, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf284
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg254_1
        buf287 = buf285; del buf285  # reuse
        # Source Nodes: [x_324, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf286, arg965_1, arg966_1, arg255_1, arg256_1, buf287, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg255_1
        del arg256_1
        del arg965_1
        del arg966_1
        del buf286
        # Source Nodes: [x_324, x_325, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf288 = extern_kernels.convolution(buf287, arg257_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf288, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg257_1
        buf289 = buf287; del buf287  # reuse
        # Source Nodes: [x_328], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf288, buf289, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf288
        # Source Nodes: [x_328], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg258_1
        del buf289
        # Source Nodes: [x_330, x_331], Original ATen: [aten.convolution, aten.relu]
        buf292 = extern_kernels.convolution(buf291, arg261_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf292, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg261_1
        buf293 = buf291; del buf291  # reuse
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf292, buf293, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf292
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg262_1
        buf295 = buf293; del buf293  # reuse
        # Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf294, arg971_1, arg972_1, arg263_1, arg264_1, buf295, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg263_1
        del arg264_1
        del arg971_1
        del arg972_1
        del buf294
        # Source Nodes: [x_334, x_335, x_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf296 = extern_kernels.convolution(buf295, arg265_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf296, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg265_1
        buf297 = buf295; del buf295  # reuse
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf296, buf297, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf296
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg266_1
        del buf297
        # Source Nodes: [x_340, x_341], Original ATen: [aten.convolution, aten.relu]
        buf300 = extern_kernels.convolution(buf299, arg269_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf300, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg269_1
        buf301 = buf299; del buf299  # reuse
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf300, buf301, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf300
        # Source Nodes: [x_343], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg270_1
        buf303 = buf301; del buf301  # reuse
        # Source Nodes: [x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf302, arg977_1, arg978_1, arg271_1, arg272_1, buf303, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg271_1
        del arg272_1
        del arg977_1
        del arg978_1
        del buf302
        # Source Nodes: [x_344, x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf304 = extern_kernels.convolution(buf303, arg273_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf304, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg273_1
        buf305 = buf303; del buf303  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf304, buf305, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf304
        # Source Nodes: [x_348], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg274_1
        buf307 = reinterpret_tensor(buf328, (8, 216, 42, 42), (1905120, 1764, 42, 1), 762048)  # alias
        buf308 = buf305; del buf305  # reuse
        # Source Nodes: [x_350, x_comb_iter_22, x_comb_iter_2_left_4, x_comb_iter_2_right_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf298, arg974_1, arg975_1, arg267_1, arg268_1, buf306, arg980_1, arg981_1, arg275_1, arg276_1, buf307, buf308, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg267_1
        del arg268_1
        del arg275_1
        del arg276_1
        del arg974_1
        del arg975_1
        del arg980_1
        del arg981_1
        del buf298
        del buf306
        # Source Nodes: [x_350, x_351], Original ATen: [aten.convolution, aten.relu]
        buf309 = extern_kernels.convolution(buf308, arg277_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf309, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg277_1
        buf310 = buf308; del buf308  # reuse
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf309, buf310, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf309
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg278_1
        buf312 = buf310; del buf310  # reuse
        # Source Nodes: [x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf311, arg983_1, arg984_1, arg279_1, arg280_1, buf312, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg279_1
        del arg280_1
        del arg983_1
        del arg984_1
        del buf311
        # Source Nodes: [x_354, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf313 = extern_kernels.convolution(buf312, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf313, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg281_1
        buf314 = buf312; del buf312  # reuse
        # Source Nodes: [x_358], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf313, buf314, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf313
        # Source Nodes: [x_358], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg282_1
        del buf314
        buf325 = reinterpret_tensor(buf328, (8, 216, 42, 42), (1905120, 1764, 42, 1), 381024)  # alias
        # Source Nodes: [x_comb_iter_1_left_4, x_comb_iter_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf290, arg968_1, arg969_1, arg259_1, arg260_1, buf269, buf325, 3048192, grid=grid(3048192), stream=stream0)
        del arg259_1
        del arg260_1
        del arg968_1
        del arg969_1
        del buf269
        del buf290
        buf326 = reinterpret_tensor(buf328, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1143072)  # alias
        # Source Nodes: [x_comb_iter_23, x_comb_iter_3_left_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf315, arg986_1, arg987_1, arg283_1, arg284_1, buf270, buf326, 3048192, grid=grid(3048192), stream=stream0)
        del arg283_1
        del arg284_1
        del arg986_1
        del arg987_1
        del buf270
        buf329 = buf266; del buf266  # reuse
        buf334 = reinterpret_tensor(buf265, (8, 1080, 42, 42), (1905120, 1, 45360, 1080), 0); del buf265  # reuse
        # Source Nodes: [x_373, x_436], Original ATen: [aten.relu]
        triton_poi_fused_relu_41.run(buf328, buf329, buf334, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del buf307
        del buf324
        del buf325
        del buf326
        del buf327
        del buf328
        # Source Nodes: [x_373, x_374], Original ATen: [aten.convolution, aten.relu]
        buf330 = extern_kernels.convolution(buf329, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg296_1
        del buf329
        buf331 = buf330; del buf330  # reuse
        # Source Nodes: [x_comb_iter_4_right_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf331, arg998_1, arg999_1, arg297_1, arg298_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg297_1
        del arg298_1
        del arg998_1
        del arg999_1
        # Source Nodes: [x_426, x_427], Original ATen: [aten.convolution, aten.relu]
        buf380 = extern_kernels.convolution(buf379, arg339_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf380, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg339_1
        buf381 = buf379; del buf379  # reuse
        # Source Nodes: [x_429], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf380, buf381, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_429], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg340_1
        buf383 = buf381; del buf381  # reuse
        # Source Nodes: [x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf382, arg1031_1, arg1032_1, arg341_1, arg342_1, buf383, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1031_1
        del arg1032_1
        del arg341_1
        del arg342_1
        # Source Nodes: [x_430, x_431, x_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf384 = extern_kernels.convolution(buf383, arg343_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf384, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg343_1
        buf385 = buf383; del buf383  # reuse
        # Source Nodes: [x_434], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf384, buf385, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        # Source Nodes: [x_434], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg344_1
        buf332 = reinterpret_tensor(buf385, (8, 216, 42, 42), (381024, 1764, 42, 1), 0); del buf385  # reuse
        buf333 = buf384; del buf384  # reuse
        buf346 = reinterpret_tensor(buf382, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf382  # reuse
        buf354 = reinterpret_tensor(buf380, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf380  # reuse
        buf362 = reinterpret_tensor(buf315, (8, 216, 42, 42), (381024, 1, 9072, 216), 0); del buf315  # reuse
        buf390 = reinterpret_tensor(buf391, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1524096)  # alias
        # Source Nodes: [x_386, x_396, x_406, x_comb_iter_1_right_5, x_comb_iter_29, x_comb_iter_3_right_5, x_comb_iter_4_left_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_37.run(buf331, buf386, arg1034_1, arg1035_1, arg345_1, arg346_1, buf332, buf333, buf346, buf354, buf362, buf390, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1034_1
        del arg1035_1
        del arg345_1
        del arg346_1
        del buf331
        del buf386
        # Source Nodes: [x_436, x_437], Original ATen: [aten.convolution, aten.relu]
        buf335 = extern_kernels.convolution(buf334, arg347_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (8, 432, 42, 42), (762048, 1764, 42, 1))
        del arg347_1
        buf336 = buf335; del buf335  # reuse
        # Source Nodes: [x_left_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf336, arg1037_1, arg1038_1, arg348_1, arg349_1, 6096384, grid=grid(6096384), stream=stream0)
        del arg1037_1
        del arg1038_1
        del arg348_1
        del arg349_1
        buf403 = empty_strided((8, 432, 45, 45), (874800, 1, 19440, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_442, x_444], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_43.run(buf336, buf403, 3456, 2025, grid=grid(3456, 2025), stream=stream0)
        # Source Nodes: [x_442, x_444, x_445], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf404 = extern_kernels.convolution(buf403, arg14_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf404, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg14_1
        buf405 = reinterpret_tensor(buf134, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf134  # reuse
        # Source Nodes: [x_447], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf404, buf405, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf404
        # Source Nodes: [x_447], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, arg353_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg353_1
        buf407 = buf405; del buf405  # reuse
        # Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf406, arg1043_1, arg1044_1, arg354_1, arg355_1, buf407, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1043_1
        del arg1044_1
        del arg354_1
        del arg355_1
        del buf406
        # Source Nodes: [x_448, x_449, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf408 = extern_kernels.convolution(buf407, arg356_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf408, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg356_1
        buf409 = buf407; del buf407  # reuse
        # Source Nodes: [x_452], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf408, buf409, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf408
        # Source Nodes: [x_452], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, arg357_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg357_1
        del buf409
        buf458 = reinterpret_tensor(buf145, (8, 2160, 21, 21), (952560, 441, 21, 1), 0); del buf145  # reuse
        buf455 = reinterpret_tensor(buf458, (8, 432, 21, 21), (952560, 441, 21, 1), 0)  # alias
        # Source Nodes: [x_455, x_comb_iter_0_left_6, x_comb_iter_0_right_6, x_comb_iter_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_46.run(buf336, buf410, arg1046_1, arg1047_1, arg358_1, arg359_1, buf455, 1524096, grid=grid(1524096), stream=stream0)
        del arg1046_1
        del arg1047_1
        del arg358_1
        del arg359_1
        # Source Nodes: [x_386, x_387], Original ATen: [aten.convolution, aten.relu]
        buf347 = extern_kernels.convolution(buf346, arg307_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf347, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg307_1
        buf348 = buf346; del buf346  # reuse
        # Source Nodes: [x_389], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf347, buf348, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf347
        # Source Nodes: [x_389], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg308_1
        buf350 = buf348; del buf348  # reuse
        # Source Nodes: [x_390, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf349, arg1007_1, arg1008_1, arg309_1, arg310_1, buf350, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1007_1
        del arg1008_1
        del arg309_1
        del arg310_1
        del buf349
        # Source Nodes: [x_390, x_391, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf351 = extern_kernels.convolution(buf350, arg311_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf351, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg311_1
        buf352 = buf350; del buf350  # reuse
        # Source Nodes: [x_394], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf351, buf352, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf351
        # Source Nodes: [x_394], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg312_1
        del buf352
        # Source Nodes: [x_396, x_397], Original ATen: [aten.convolution, aten.relu]
        buf355 = extern_kernels.convolution(buf354, arg315_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf355, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg315_1
        buf356 = buf354; del buf354  # reuse
        # Source Nodes: [x_399], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf355, buf356, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf355
        # Source Nodes: [x_399], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf356, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg316_1
        buf358 = buf356; del buf356  # reuse
        # Source Nodes: [x_400, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf357, arg1013_1, arg1014_1, arg317_1, arg318_1, buf358, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1013_1
        del arg1014_1
        del arg317_1
        del arg318_1
        del buf357
        # Source Nodes: [x_400, x_401, x_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf359 = extern_kernels.convolution(buf358, arg319_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf359, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg319_1
        buf360 = buf358; del buf358  # reuse
        # Source Nodes: [x_404], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf359, buf360, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf359
        # Source Nodes: [x_404], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg320_1
        del buf360
        # Source Nodes: [x_406, x_407], Original ATen: [aten.convolution, aten.relu]
        buf363 = extern_kernels.convolution(buf362, arg323_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf363, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg323_1
        buf364 = buf362; del buf362  # reuse
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf363, buf364, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf363
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg324_1
        buf366 = buf364; del buf364  # reuse
        # Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf365, arg1019_1, arg1020_1, arg325_1, arg326_1, buf366, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1019_1
        del arg1020_1
        del arg325_1
        del arg326_1
        del buf365
        # Source Nodes: [x_410, x_411, x_412], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf367 = extern_kernels.convolution(buf366, arg327_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf367, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg327_1
        buf368 = buf366; del buf366  # reuse
        # Source Nodes: [x_414], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf367, buf368, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf367
        # Source Nodes: [x_414], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, arg328_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg328_1
        buf370 = reinterpret_tensor(buf391, (8, 216, 42, 42), (1905120, 1764, 42, 1), 762048)  # alias
        buf371 = buf368; del buf368  # reuse
        # Source Nodes: [x_416, x_comb_iter_27, x_comb_iter_2_left_5, x_comb_iter_2_right_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf361, arg1016_1, arg1017_1, arg321_1, arg322_1, buf369, arg1022_1, arg1023_1, arg329_1, arg330_1, buf370, buf371, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1016_1
        del arg1017_1
        del arg1022_1
        del arg1023_1
        del arg321_1
        del arg322_1
        del arg329_1
        del arg330_1
        del buf361
        del buf369
        # Source Nodes: [x_416, x_417], Original ATen: [aten.convolution, aten.relu]
        buf372 = extern_kernels.convolution(buf371, arg331_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf372, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg331_1
        buf373 = buf371; del buf371  # reuse
        # Source Nodes: [x_419], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf372, buf373, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf372
        # Source Nodes: [x_419], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg332_1
        buf375 = buf373; del buf373  # reuse
        # Source Nodes: [x_420, x_421], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf374, arg1025_1, arg1026_1, arg333_1, arg334_1, buf375, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg1025_1
        del arg1026_1
        del arg333_1
        del arg334_1
        del buf374
        # Source Nodes: [x_420, x_421, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf376 = extern_kernels.convolution(buf375, arg335_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf376, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg335_1
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf376, buf377, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del buf376
        # Source Nodes: [x_424], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (8, 216, 42, 42), (381024, 1764, 42, 1))
        del arg336_1
        del buf377
        buf388 = reinterpret_tensor(buf391, (8, 216, 42, 42), (1905120, 1764, 42, 1), 381024)  # alias
        # Source Nodes: [x_comb_iter_1_left_5, x_comb_iter_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf353, arg1010_1, arg1011_1, arg313_1, arg314_1, buf332, buf388, 3048192, grid=grid(3048192), stream=stream0)
        del arg1010_1
        del arg1011_1
        del arg313_1
        del arg314_1
        del buf332
        del buf353
        buf389 = reinterpret_tensor(buf391, (8, 216, 42, 42), (1905120, 1764, 42, 1), 1143072)  # alias
        # Source Nodes: [x_comb_iter_28, x_comb_iter_3_left_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf378, arg1028_1, arg1029_1, arg337_1, arg338_1, buf333, buf389, 3048192, grid=grid(3048192), stream=stream0)
        del arg1028_1
        del arg1029_1
        del arg337_1
        del arg338_1
        del buf333
        buf392 = buf334; del buf334  # reuse
        # Source Nodes: [x_439], Original ATen: [aten.relu]
        triton_poi_fused_relu_47.run(buf391, buf392, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del buf370
        del buf387
        del buf388
        del buf389
        del buf390
        # Source Nodes: [x_439, x_440], Original ATen: [aten.convolution, aten.relu]
        buf393 = extern_kernels.convolution(buf392, arg350_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 432, 42, 42), (762048, 1764, 42, 1))
        del arg350_1
        del buf392
        buf394 = buf393; del buf393  # reuse
        buf452 = empty_strided((8, 432, 42, 42), (762048, 1, 18144, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_518, x_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf394, arg1040_1, arg1041_1, arg351_1, arg352_1, buf452, 3456, 1764, grid=grid(3456, 1764), stream=stream0)
        del arg1040_1
        del arg1041_1
        del arg351_1
        del arg352_1
        buf411 = empty_strided((8, 432, 47, 47), (954288, 1, 20304, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_456, x_458], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_49.run(buf394, buf411, 3456, 2209, grid=grid(3456, 2209), stream=stream0)
        # Source Nodes: [x_456, x_458, x_459], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf412 = extern_kernels.convolution(buf411, arg15_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf412, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg15_1
        del buf411
        buf413 = reinterpret_tensor(buf410, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf410  # reuse
        # Source Nodes: [x_461], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf412, buf413, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf412
        # Source Nodes: [x_461], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, arg360_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg360_1
        buf415 = buf413; del buf413  # reuse
        # Source Nodes: [x_462, x_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf414, arg1049_1, arg1050_1, arg361_1, arg362_1, buf415, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1049_1
        del arg1050_1
        del arg361_1
        del arg362_1
        del buf414
        # Source Nodes: [x_462, x_463, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf416 = extern_kernels.convolution(buf415, arg363_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf416, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg363_1
        buf417 = buf415; del buf415  # reuse
        # Source Nodes: [x_466], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf416, buf417, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf416
        # Source Nodes: [x_466], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, arg364_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg364_1
        buf419 = buf403; del buf403  # reuse
        # Source Nodes: [x_470, x_472], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_43.run(buf394, buf419, 3456, 2025, grid=grid(3456, 2025), stream=stream0)
        # Source Nodes: [x_470, x_472, x_473], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf420 = extern_kernels.convolution(buf419, arg16_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf420, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg16_1
        del buf419
        buf421 = buf417; del buf417  # reuse
        # Source Nodes: [x_475], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf420, buf421, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf420
        # Source Nodes: [x_475], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, arg367_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg367_1
        buf423 = buf421; del buf421  # reuse
        # Source Nodes: [x_476, x_477], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf422, arg1055_1, arg1056_1, arg368_1, arg369_1, buf423, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1055_1
        del arg1056_1
        del arg368_1
        del arg369_1
        del buf422
        # Source Nodes: [x_476, x_477, x_478], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf424 = extern_kernels.convolution(buf423, arg370_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf424, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg370_1
        buf425 = buf423; del buf423  # reuse
        # Source Nodes: [x_480], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf424, buf425, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf424
        # Source Nodes: [x_480], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, arg371_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg371_1
        buf427 = empty_strided((8, 432, 43, 43), (798768, 1, 18576, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_482, x_484], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_50.run(buf394, buf427, 3456, 1849, grid=grid(3456, 1849), stream=stream0)
        # Source Nodes: [x_482, x_484, x_485], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf428 = extern_kernels.convolution(buf427, arg17_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf428, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg17_1
        buf429 = buf425; del buf425  # reuse
        # Source Nodes: [x_487], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf428, buf429, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf428
        # Source Nodes: [x_487], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, arg374_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg374_1
        buf431 = buf429; del buf429  # reuse
        # Source Nodes: [x_488, x_489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf430, arg1061_1, arg1062_1, arg375_1, arg376_1, buf431, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1061_1
        del arg1062_1
        del arg375_1
        del arg376_1
        del buf430
        # Source Nodes: [x_488, x_489, x_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf432 = extern_kernels.convolution(buf431, arg377_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf432, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg377_1
        buf433 = buf431; del buf431  # reuse
        # Source Nodes: [x_492], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf432, buf433, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf432
        # Source Nodes: [x_492], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, arg378_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg378_1
        buf435 = reinterpret_tensor(buf458, (8, 432, 21, 21), (952560, 441, 21, 1), 381024)  # alias
        buf436 = buf433; del buf433  # reuse
        # Source Nodes: [x_494, x_comb_iter_2_left_6, x_comb_iter_2_right_6, x_comb_iter_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51.run(buf426, arg1058_1, arg1059_1, arg372_1, arg373_1, buf434, arg1064_1, arg1065_1, arg379_1, arg380_1, buf435, buf436, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1058_1
        del arg1059_1
        del arg1064_1
        del arg1065_1
        del arg372_1
        del arg373_1
        del arg379_1
        del arg380_1
        del buf426
        del buf434
        # Source Nodes: [x_494, x_495], Original ATen: [aten.convolution, aten.relu]
        buf437 = extern_kernels.convolution(buf436, arg381_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf437, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg381_1
        buf438 = buf436; del buf436  # reuse
        # Source Nodes: [x_497], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf437, buf438, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf437
        # Source Nodes: [x_497], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, arg382_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg382_1
        buf440 = buf438; del buf438  # reuse
        # Source Nodes: [x_498, x_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf439, arg1067_1, arg1068_1, arg383_1, arg384_1, buf440, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1067_1
        del arg1068_1
        del arg383_1
        del arg384_1
        del buf439
        # Source Nodes: [x_498, x_499, x_500], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf441 = extern_kernels.convolution(buf440, arg385_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf441, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg385_1
        buf442 = buf440; del buf440  # reuse
        # Source Nodes: [x_502], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf441, buf442, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf441
        # Source Nodes: [x_502], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, arg386_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg386_1
        buf456 = reinterpret_tensor(buf458, (8, 432, 21, 21), (952560, 441, 21, 1), 190512)  # alias
        buf457 = reinterpret_tensor(buf458, (8, 432, 21, 21), (952560, 441, 21, 1), 571536)  # alias
        # Source Nodes: [x_469, x_505, x_comb_iter_1_left_6, x_comb_iter_1_right_6, x_comb_iter_31, x_comb_iter_33, x_comb_iter_3_left_6, x_comb_iter_3_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_52.run(buf394, buf418, arg1052_1, arg1053_1, arg365_1, arg366_1, buf443, arg1070_1, arg1071_1, arg387_1, arg388_1, buf456, buf457, 1524096, grid=grid(1524096), stream=stream0)
        del arg1052_1
        del arg1053_1
        del arg1070_1
        del arg1071_1
        del arg365_1
        del arg366_1
        del arg387_1
        del arg388_1
        del buf394
        buf397 = reinterpret_tensor(buf80, (8, 1080, 21, 21), (476280, 1, 22680, 1080), 0); del buf80  # reuse
        # Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_1_avgpool, x_523], Original ATen: [aten.avg_pool2d, aten.relu]
        triton_poi_fused_avg_pool2d_relu_53.run(buf391, buf397, 8640, 441, grid=grid(8640, 441), stream=stream0)
        # Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_1_avgpool, x_523, x_path1_2], Original ATen: [aten.avg_pool2d, aten.convolution, aten.relu]
        buf398 = extern_kernels.convolution(buf397, arg398_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 216, 21, 21), (95256, 441, 21, 1))
        del arg398_1
        buf399 = buf397; del buf397  # reuse
        # Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_2_avgpool, l__mod___cell_5_conv_prev_1x1_path_2_pad, x_523], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_54.run(buf391, buf399, 8640, 441, grid=grid(8640, 441), stream=stream0)
        del buf391
        # Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_2_avgpool, l__mod___cell_5_conv_prev_1x1_path_2_pad, x_523, x_path2_2], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.relu]
        buf400 = extern_kernels.convolution(buf399, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (8, 216, 21, 21), (95256, 441, 21, 1))
        del arg399_1
        del buf399
        buf401 = buf443; del buf443  # reuse
        buf468 = reinterpret_tensor(buf418, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf418  # reuse
        buf509 = buf442; del buf442  # reuse
        # Source Nodes: [cat_26, x_527, x_577, x_left_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55.run(buf398, buf400, arg1082_1, arg1083_1, arg400_1, arg401_1, buf401, buf468, buf509, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1082_1
        del arg1083_1
        del arg400_1
        del arg401_1
        del buf398
        del buf400
        # Source Nodes: [x_527, x_528], Original ATen: [aten.convolution, aten.relu]
        buf469 = extern_kernels.convolution(buf468, arg405_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf469, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg405_1
        buf470 = buf468; del buf468  # reuse
        # Source Nodes: [x_530], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf469, buf470, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf469
        # Source Nodes: [x_530], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, arg406_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg406_1
        buf472 = buf470; del buf470  # reuse
        # Source Nodes: [x_531, x_532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf471, arg1088_1, arg1089_1, arg407_1, arg408_1, buf472, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1088_1
        del arg1089_1
        del arg407_1
        del arg408_1
        del buf471
        # Source Nodes: [x_531, x_532, x_533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf473 = extern_kernels.convolution(buf472, arg409_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf473, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg409_1
        buf474 = buf472; del buf472  # reuse
        # Source Nodes: [x_535], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf473, buf474, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf473
        # Source Nodes: [x_535], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, arg410_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg410_1
        del buf474
        buf521 = reinterpret_tensor(buf140, (8, 2160, 21, 21), (952560, 441, 21, 1), 0); del buf140  # reuse
        buf517 = reinterpret_tensor(buf521, (8, 432, 21, 21), (952560, 441, 21, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_7, x_comb_iter_0_right_7, x_comb_iter_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_56.run(buf401, buf475, arg1091_1, arg1092_1, arg411_1, arg412_1, buf517, 1524096, grid=grid(1524096), stream=stream0)
        del arg1091_1
        del arg1092_1
        del arg411_1
        del arg412_1
        del buf401
        buf444 = buf427; del buf427  # reuse
        # Source Nodes: [x_506, x_508], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_50.run(buf336, buf444, 3456, 1849, grid=grid(3456, 1849), stream=stream0)
        del buf336
        # Source Nodes: [x_506, x_508, x_509], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf445 = extern_kernels.convolution(buf444, arg18_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf445, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg18_1
        del buf444
        buf446 = reinterpret_tensor(buf475, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf475  # reuse
        # Source Nodes: [x_511], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf445, buf446, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf445
        # Source Nodes: [x_511], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, arg389_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg389_1
        buf448 = buf446; del buf446  # reuse
        # Source Nodes: [x_512, x_513], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf447, arg1073_1, arg1074_1, arg390_1, arg391_1, buf448, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1073_1
        del arg1074_1
        del arg390_1
        del arg391_1
        del buf447
        # Source Nodes: [x_512, x_513, x_514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf449 = extern_kernels.convolution(buf448, arg392_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf449, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg392_1
        buf450 = buf448; del buf448  # reuse
        # Source Nodes: [x_516], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf449, buf450, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf449
        # Source Nodes: [x_516], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, arg393_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg393_1
        del buf450
        # Source Nodes: [x_518, x_521], Original ATen: [aten.convolution, aten.relu]
        buf453 = extern_kernels.convolution(buf452, arg19_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg19_1
        del buf452
        buf454 = reinterpret_tensor(buf458, (8, 432, 21, 21), (952560, 441, 21, 1), 762048)  # alias
        # Source Nodes: [x_comb_iter_34, x_comb_iter_4_left_6, x_comb_iter_4_right_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf451, arg1076_1, arg1077_1, arg394_1, arg395_1, buf453, arg1079_1, arg1080_1, arg396_1, arg397_1, buf454, 1524096, grid=grid(1524096), stream=stream0)
        del arg1076_1
        del arg1077_1
        del arg1079_1
        del arg1080_1
        del arg394_1
        del arg395_1
        del arg396_1
        del arg397_1
        del buf451
        buf459 = reinterpret_tensor(buf139, (8, 2160, 21, 21), (952560, 1, 45360, 2160), 0); del buf139  # reuse
        buf464 = empty_strided((8, 2160, 21, 21), (952560, 1, 45360, 2160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_524, x_587], Original ATen: [aten.relu]
        triton_poi_fused_relu_58.run(buf458, buf459, buf464, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del buf435
        del buf454
        del buf455
        del buf456
        del buf457
        # Source Nodes: [x_524, x_525], Original ATen: [aten.convolution, aten.relu]
        buf460 = extern_kernels.convolution(buf459, arg402_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg402_1
        buf461 = buf460; del buf460  # reuse
        # Source Nodes: [x_comb_iter_4_right_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_59.run(buf461, arg1085_1, arg1086_1, arg403_1, arg404_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg1085_1
        del arg1086_1
        del arg403_1
        del arg404_1
        # Source Nodes: [x_577, x_578], Original ATen: [aten.convolution, aten.relu]
        buf510 = extern_kernels.convolution(buf509, arg445_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf510, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg445_1
        buf511 = buf509; del buf509  # reuse
        # Source Nodes: [x_580], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf510, buf511, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_580], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, arg446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg446_1
        buf513 = buf511; del buf511  # reuse
        # Source Nodes: [x_581, x_582], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf512, arg1118_1, arg1119_1, arg447_1, arg448_1, buf513, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1118_1
        del arg1119_1
        del arg447_1
        del arg448_1
        # Source Nodes: [x_581, x_582, x_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf514 = extern_kernels.convolution(buf513, arg449_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf514, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg449_1
        buf515 = buf513; del buf513  # reuse
        # Source Nodes: [x_585], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf514, buf515, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_585], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, arg450_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg450_1
        buf462 = reinterpret_tensor(buf515, (8, 432, 21, 21), (190512, 441, 21, 1), 0); del buf515  # reuse
        buf463 = buf514; del buf514  # reuse
        buf476 = reinterpret_tensor(buf512, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf512  # reuse
        buf484 = reinterpret_tensor(buf510, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf510  # reuse
        buf492 = reinterpret_tensor(buf453, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf453  # reuse
        buf520 = reinterpret_tensor(buf521, (8, 432, 21, 21), (952560, 441, 21, 1), 762048)  # alias
        # Source Nodes: [x_537, x_547, x_557, x_comb_iter_1_right_7, x_comb_iter_39, x_comb_iter_3_right_7, x_comb_iter_4_left_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_60.run(buf461, buf516, arg1121_1, arg1122_1, arg451_1, arg452_1, buf462, buf463, buf476, buf484, buf492, buf520, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1121_1
        del arg1122_1
        del arg451_1
        del arg452_1
        # Source Nodes: [x_587, x_588], Original ATen: [aten.convolution, aten.relu]
        buf465 = extern_kernels.convolution(buf464, arg453_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg453_1
        buf466 = buf465; del buf465  # reuse
        buf531 = reinterpret_tensor(buf516, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf516  # reuse
        buf572 = reinterpret_tensor(buf461, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf461  # reuse
        # Source Nodes: [x_593, x_643, x_left_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf466, arg1124_1, arg1125_1, arg454_1, arg455_1, buf531, buf572, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1124_1
        del arg1125_1
        del arg454_1
        del arg455_1
        # Source Nodes: [x_593, x_594], Original ATen: [aten.convolution, aten.relu]
        buf532 = extern_kernels.convolution(buf531, arg459_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf532, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg459_1
        buf533 = buf531; del buf531  # reuse
        # Source Nodes: [x_596], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf532, buf533, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf532
        # Source Nodes: [x_596], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, arg460_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg460_1
        buf535 = buf533; del buf533  # reuse
        # Source Nodes: [x_597, x_598], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf534, arg1130_1, arg1131_1, arg461_1, arg462_1, buf535, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1130_1
        del arg1131_1
        del arg461_1
        del arg462_1
        del buf534
        # Source Nodes: [x_597, x_598, x_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf536 = extern_kernels.convolution(buf535, arg463_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf536, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg463_1
        buf537 = buf535; del buf535  # reuse
        # Source Nodes: [x_601], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf536, buf537, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf536
        # Source Nodes: [x_601], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, arg464_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg464_1
        del buf537
        buf584 = reinterpret_tensor(buf464, (8, 2160, 21, 21), (952560, 441, 21, 1), 0); del buf464  # reuse
        buf580 = reinterpret_tensor(buf584, (8, 432, 21, 21), (952560, 441, 21, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_8, x_comb_iter_0_right_8, x_comb_iter_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_56.run(buf466, buf538, arg1133_1, arg1134_1, arg465_1, arg466_1, buf580, 1524096, grid=grid(1524096), stream=stream0)
        del arg1133_1
        del arg1134_1
        del arg465_1
        del arg466_1
        del buf466
        del buf538
        # Source Nodes: [x_537, x_538], Original ATen: [aten.convolution, aten.relu]
        buf477 = extern_kernels.convolution(buf476, arg413_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf477, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg413_1
        buf478 = buf476; del buf476  # reuse
        # Source Nodes: [x_540], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf477, buf478, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf477
        # Source Nodes: [x_540], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, arg414_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg414_1
        buf480 = buf478; del buf478  # reuse
        # Source Nodes: [x_541, x_542], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf479, arg1094_1, arg1095_1, arg415_1, arg416_1, buf480, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1094_1
        del arg1095_1
        del arg415_1
        del arg416_1
        del buf479
        # Source Nodes: [x_541, x_542, x_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf481 = extern_kernels.convolution(buf480, arg417_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf481, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg417_1
        buf482 = buf480; del buf480  # reuse
        # Source Nodes: [x_545], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf481, buf482, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf481
        # Source Nodes: [x_545], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, arg418_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg418_1
        del buf482
        # Source Nodes: [x_547, x_548], Original ATen: [aten.convolution, aten.relu]
        buf485 = extern_kernels.convolution(buf484, arg421_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf485, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg421_1
        buf486 = buf484; del buf484  # reuse
        # Source Nodes: [x_550], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf485, buf486, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf485
        # Source Nodes: [x_550], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, arg422_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg422_1
        buf488 = buf486; del buf486  # reuse
        # Source Nodes: [x_551, x_552], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf487, arg1100_1, arg1101_1, arg423_1, arg424_1, buf488, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1100_1
        del arg1101_1
        del arg423_1
        del arg424_1
        del buf487
        # Source Nodes: [x_551, x_552, x_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf489 = extern_kernels.convolution(buf488, arg425_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf489, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg425_1
        buf490 = buf488; del buf488  # reuse
        # Source Nodes: [x_555], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf489, buf490, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf489
        # Source Nodes: [x_555], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, arg426_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg426_1
        del buf490
        # Source Nodes: [x_557, x_558], Original ATen: [aten.convolution, aten.relu]
        buf493 = extern_kernels.convolution(buf492, arg429_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf493, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg429_1
        buf494 = buf492; del buf492  # reuse
        # Source Nodes: [x_560], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf493, buf494, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf493
        # Source Nodes: [x_560], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, arg430_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg430_1
        buf496 = buf494; del buf494  # reuse
        # Source Nodes: [x_561, x_562], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf495, arg1106_1, arg1107_1, arg431_1, arg432_1, buf496, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1106_1
        del arg1107_1
        del arg431_1
        del arg432_1
        del buf495
        # Source Nodes: [x_561, x_562, x_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf497 = extern_kernels.convolution(buf496, arg433_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf497, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg433_1
        buf498 = buf496; del buf496  # reuse
        # Source Nodes: [x_565], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf497, buf498, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf497
        # Source Nodes: [x_565], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, arg434_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg434_1
        buf500 = reinterpret_tensor(buf521, (8, 432, 21, 21), (952560, 441, 21, 1), 381024)  # alias
        buf501 = buf498; del buf498  # reuse
        # Source Nodes: [x_567, x_comb_iter_2_left_7, x_comb_iter_2_right_7, x_comb_iter_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51.run(buf491, arg1103_1, arg1104_1, arg427_1, arg428_1, buf499, arg1109_1, arg1110_1, arg435_1, arg436_1, buf500, buf501, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1103_1
        del arg1104_1
        del arg1109_1
        del arg1110_1
        del arg427_1
        del arg428_1
        del arg435_1
        del arg436_1
        del buf491
        del buf499
        # Source Nodes: [x_567, x_568], Original ATen: [aten.convolution, aten.relu]
        buf502 = extern_kernels.convolution(buf501, arg437_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf502, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg437_1
        buf503 = buf501; del buf501  # reuse
        # Source Nodes: [x_570], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf502, buf503, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf502
        # Source Nodes: [x_570], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, arg438_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg438_1
        buf505 = buf503; del buf503  # reuse
        # Source Nodes: [x_571, x_572], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf504, arg1112_1, arg1113_1, arg439_1, arg440_1, buf505, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1112_1
        del arg1113_1
        del arg439_1
        del arg440_1
        del buf504
        # Source Nodes: [x_571, x_572, x_573], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf506 = extern_kernels.convolution(buf505, arg441_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf506, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg441_1
        buf507 = buf505; del buf505  # reuse
        # Source Nodes: [x_575], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf506, buf507, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf506
        # Source Nodes: [x_575], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, arg442_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg442_1
        del buf507
        buf518 = reinterpret_tensor(buf521, (8, 432, 21, 21), (952560, 441, 21, 1), 190512)  # alias
        # Source Nodes: [x_comb_iter_1_left_7, x_comb_iter_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf483, arg1097_1, arg1098_1, arg419_1, arg420_1, buf462, buf518, 1524096, grid=grid(1524096), stream=stream0)
        del arg1097_1
        del arg1098_1
        del arg419_1
        del arg420_1
        del buf462
        del buf483
        buf519 = reinterpret_tensor(buf521, (8, 432, 21, 21), (952560, 441, 21, 1), 571536)  # alias
        # Source Nodes: [x_comb_iter_38, x_comb_iter_3_left_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf508, arg1115_1, arg1116_1, arg443_1, arg444_1, buf463, buf519, 1524096, grid=grid(1524096), stream=stream0)
        del arg1115_1
        del arg1116_1
        del arg443_1
        del arg444_1
        del buf463
        buf522 = buf459; del buf459  # reuse
        buf527 = reinterpret_tensor(buf458, (8, 2160, 21, 21), (952560, 1, 45360, 2160), 0); del buf458  # reuse
        # Source Nodes: [x_590, x_653], Original ATen: [aten.relu]
        triton_poi_fused_relu_58.run(buf521, buf522, buf527, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del buf500
        del buf517
        del buf518
        del buf519
        del buf520
        # Source Nodes: [x_590, x_591], Original ATen: [aten.convolution, aten.relu]
        buf523 = extern_kernels.convolution(buf522, arg456_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg456_1
        buf524 = buf523; del buf523  # reuse
        # Source Nodes: [x_comb_iter_4_right_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_59.run(buf524, arg1127_1, arg1128_1, arg457_1, arg458_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg1127_1
        del arg1128_1
        del arg457_1
        del arg458_1
        # Source Nodes: [x_643, x_644], Original ATen: [aten.convolution, aten.relu]
        buf573 = extern_kernels.convolution(buf572, arg499_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf573, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg499_1
        buf574 = buf572; del buf572  # reuse
        # Source Nodes: [x_646], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf573, buf574, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_646], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, arg500_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg500_1
        buf576 = buf574; del buf574  # reuse
        # Source Nodes: [x_647, x_648], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf575, arg1160_1, arg1161_1, arg501_1, arg502_1, buf576, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1160_1
        del arg1161_1
        del arg501_1
        del arg502_1
        # Source Nodes: [x_647, x_648, x_649], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf577 = extern_kernels.convolution(buf576, arg503_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf577, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg503_1
        buf578 = buf576; del buf576  # reuse
        # Source Nodes: [x_651], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf577, buf578, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_651], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, arg504_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg504_1
        buf525 = reinterpret_tensor(buf578, (8, 432, 21, 21), (190512, 441, 21, 1), 0); del buf578  # reuse
        buf526 = buf577; del buf577  # reuse
        buf539 = reinterpret_tensor(buf575, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf575  # reuse
        buf547 = reinterpret_tensor(buf573, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf573  # reuse
        buf555 = reinterpret_tensor(buf508, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf508  # reuse
        buf583 = reinterpret_tensor(buf584, (8, 432, 21, 21), (952560, 441, 21, 1), 762048)  # alias
        # Source Nodes: [x_603, x_613, x_623, x_comb_iter_1_right_8, x_comb_iter_3_right_8, x_comb_iter_44, x_comb_iter_4_left_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_60.run(buf524, buf579, arg1163_1, arg1164_1, arg505_1, arg506_1, buf525, buf526, buf539, buf547, buf555, buf583, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1163_1
        del arg1164_1
        del arg505_1
        del arg506_1
        # Source Nodes: [x_653, x_654], Original ATen: [aten.convolution, aten.relu]
        buf528 = extern_kernels.convolution(buf527, arg507_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg507_1
        buf529 = buf528; del buf528  # reuse
        buf594 = reinterpret_tensor(buf579, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf579  # reuse
        buf635 = reinterpret_tensor(buf524, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf524  # reuse
        # Source Nodes: [x_659, x_709, x_left_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf529, arg1166_1, arg1167_1, arg508_1, arg509_1, buf594, buf635, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1166_1
        del arg1167_1
        del arg508_1
        del arg509_1
        # Source Nodes: [x_659, x_660], Original ATen: [aten.convolution, aten.relu]
        buf595 = extern_kernels.convolution(buf594, arg513_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf595, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg513_1
        buf596 = buf594; del buf594  # reuse
        # Source Nodes: [x_662], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf595, buf596, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf595
        # Source Nodes: [x_662], Original ATen: [aten.convolution]
        buf597 = extern_kernels.convolution(buf596, arg514_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg514_1
        buf598 = buf596; del buf596  # reuse
        # Source Nodes: [x_663, x_664], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf597, arg1172_1, arg1173_1, arg515_1, arg516_1, buf598, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1172_1
        del arg1173_1
        del arg515_1
        del arg516_1
        del buf597
        # Source Nodes: [x_663, x_664, x_665], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf599 = extern_kernels.convolution(buf598, arg517_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf599, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg517_1
        buf600 = buf598; del buf598  # reuse
        # Source Nodes: [x_667], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf599, buf600, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf599
        # Source Nodes: [x_667], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, arg518_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg518_1
        del buf600
        buf647 = reinterpret_tensor(buf527, (8, 2160, 21, 21), (952560, 441, 21, 1), 0); del buf527  # reuse
        buf643 = reinterpret_tensor(buf647, (8, 432, 21, 21), (952560, 441, 21, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_9, x_comb_iter_0_right_9, x_comb_iter_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_56.run(buf529, buf601, arg1175_1, arg1176_1, arg519_1, arg520_1, buf643, 1524096, grid=grid(1524096), stream=stream0)
        del arg1175_1
        del arg1176_1
        del arg519_1
        del arg520_1
        del buf529
        del buf601
        # Source Nodes: [x_603, x_604], Original ATen: [aten.convolution, aten.relu]
        buf540 = extern_kernels.convolution(buf539, arg467_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf540, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg467_1
        buf541 = buf539; del buf539  # reuse
        # Source Nodes: [x_606], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf540, buf541, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf540
        # Source Nodes: [x_606], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, arg468_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg468_1
        buf543 = buf541; del buf541  # reuse
        # Source Nodes: [x_607, x_608], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf542, arg1136_1, arg1137_1, arg469_1, arg470_1, buf543, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1136_1
        del arg1137_1
        del arg469_1
        del arg470_1
        del buf542
        # Source Nodes: [x_607, x_608, x_609], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf544 = extern_kernels.convolution(buf543, arg471_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf544, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg471_1
        buf545 = buf543; del buf543  # reuse
        # Source Nodes: [x_611], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf544, buf545, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf544
        # Source Nodes: [x_611], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, arg472_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg472_1
        del buf545
        # Source Nodes: [x_613, x_614], Original ATen: [aten.convolution, aten.relu]
        buf548 = extern_kernels.convolution(buf547, arg475_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf548, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg475_1
        buf549 = buf547; del buf547  # reuse
        # Source Nodes: [x_616], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf548, buf549, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf548
        # Source Nodes: [x_616], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, arg476_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg476_1
        buf551 = buf549; del buf549  # reuse
        # Source Nodes: [x_617, x_618], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf550, arg1142_1, arg1143_1, arg477_1, arg478_1, buf551, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1142_1
        del arg1143_1
        del arg477_1
        del arg478_1
        del buf550
        # Source Nodes: [x_617, x_618, x_619], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf552 = extern_kernels.convolution(buf551, arg479_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf552, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg479_1
        buf553 = buf551; del buf551  # reuse
        # Source Nodes: [x_621], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf552, buf553, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf552
        # Source Nodes: [x_621], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, arg480_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg480_1
        del buf553
        # Source Nodes: [x_623, x_624], Original ATen: [aten.convolution, aten.relu]
        buf556 = extern_kernels.convolution(buf555, arg483_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf556, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg483_1
        buf557 = buf555; del buf555  # reuse
        # Source Nodes: [x_626], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf556, buf557, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf556
        # Source Nodes: [x_626], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, arg484_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg484_1
        buf559 = buf557; del buf557  # reuse
        # Source Nodes: [x_627, x_628], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf558, arg1148_1, arg1149_1, arg485_1, arg486_1, buf559, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1148_1
        del arg1149_1
        del arg485_1
        del arg486_1
        del buf558
        # Source Nodes: [x_627, x_628, x_629], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf560 = extern_kernels.convolution(buf559, arg487_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf560, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg487_1
        buf561 = buf559; del buf559  # reuse
        # Source Nodes: [x_631], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf560, buf561, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf560
        # Source Nodes: [x_631], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, arg488_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg488_1
        buf563 = reinterpret_tensor(buf584, (8, 432, 21, 21), (952560, 441, 21, 1), 381024)  # alias
        buf564 = buf561; del buf561  # reuse
        # Source Nodes: [x_633, x_comb_iter_2_left_8, x_comb_iter_2_right_8, x_comb_iter_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51.run(buf554, arg1145_1, arg1146_1, arg481_1, arg482_1, buf562, arg1151_1, arg1152_1, arg489_1, arg490_1, buf563, buf564, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1145_1
        del arg1146_1
        del arg1151_1
        del arg1152_1
        del arg481_1
        del arg482_1
        del arg489_1
        del arg490_1
        del buf554
        del buf562
        # Source Nodes: [x_633, x_634], Original ATen: [aten.convolution, aten.relu]
        buf565 = extern_kernels.convolution(buf564, arg491_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf565, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg491_1
        buf566 = buf564; del buf564  # reuse
        # Source Nodes: [x_636], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf565, buf566, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf565
        # Source Nodes: [x_636], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, arg492_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg492_1
        buf568 = buf566; del buf566  # reuse
        # Source Nodes: [x_637, x_638], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf567, arg1154_1, arg1155_1, arg493_1, arg494_1, buf568, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1154_1
        del arg1155_1
        del arg493_1
        del arg494_1
        del buf567
        # Source Nodes: [x_637, x_638, x_639], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf569 = extern_kernels.convolution(buf568, arg495_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf569, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg495_1
        buf570 = buf568; del buf568  # reuse
        # Source Nodes: [x_641], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf569, buf570, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf569
        # Source Nodes: [x_641], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, arg496_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg496_1
        del buf570
        buf581 = reinterpret_tensor(buf584, (8, 432, 21, 21), (952560, 441, 21, 1), 190512)  # alias
        # Source Nodes: [x_comb_iter_1_left_8, x_comb_iter_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf546, arg1139_1, arg1140_1, arg473_1, arg474_1, buf525, buf581, 1524096, grid=grid(1524096), stream=stream0)
        del arg1139_1
        del arg1140_1
        del arg473_1
        del arg474_1
        del buf525
        del buf546
        buf582 = reinterpret_tensor(buf584, (8, 432, 21, 21), (952560, 441, 21, 1), 571536)  # alias
        # Source Nodes: [x_comb_iter_3_left_8, x_comb_iter_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf571, arg1157_1, arg1158_1, arg497_1, arg498_1, buf526, buf582, 1524096, grid=grid(1524096), stream=stream0)
        del arg1157_1
        del arg1158_1
        del arg497_1
        del arg498_1
        del buf526
        buf585 = buf522; del buf522  # reuse
        buf590 = reinterpret_tensor(buf521, (8, 2160, 21, 21), (952560, 1, 45360, 2160), 0); del buf521  # reuse
        # Source Nodes: [x_656, x_719], Original ATen: [aten.relu]
        triton_poi_fused_relu_58.run(buf584, buf585, buf590, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del buf563
        del buf580
        del buf581
        del buf582
        del buf583
        del buf584
        # Source Nodes: [x_656, x_657], Original ATen: [aten.convolution, aten.relu]
        buf586 = extern_kernels.convolution(buf585, arg510_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg510_1
        del buf585
        buf587 = buf586; del buf586  # reuse
        # Source Nodes: [x_comb_iter_4_right_9], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_59.run(buf587, arg1169_1, arg1170_1, arg511_1, arg512_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg1169_1
        del arg1170_1
        del arg511_1
        del arg512_1
        # Source Nodes: [x_709, x_710], Original ATen: [aten.convolution, aten.relu]
        buf636 = extern_kernels.convolution(buf635, arg553_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf636, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg553_1
        buf637 = buf635; del buf635  # reuse
        # Source Nodes: [x_712], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf636, buf637, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_712], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, arg554_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg554_1
        buf639 = buf637; del buf637  # reuse
        # Source Nodes: [x_713, x_714], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf638, arg1202_1, arg1203_1, arg555_1, arg556_1, buf639, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1202_1
        del arg1203_1
        del arg555_1
        del arg556_1
        # Source Nodes: [x_713, x_714, x_715], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf640 = extern_kernels.convolution(buf639, arg557_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf640, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg557_1
        buf641 = buf639; del buf639  # reuse
        # Source Nodes: [x_717], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf640, buf641, 3456, 441, grid=grid(3456, 441), stream=stream0)
        # Source Nodes: [x_717], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, arg558_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg558_1
        buf588 = reinterpret_tensor(buf641, (8, 432, 21, 21), (190512, 441, 21, 1), 0); del buf641  # reuse
        buf589 = buf640; del buf640  # reuse
        buf602 = reinterpret_tensor(buf638, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf638  # reuse
        buf610 = reinterpret_tensor(buf636, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf636  # reuse
        buf618 = reinterpret_tensor(buf571, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf571  # reuse
        buf646 = reinterpret_tensor(buf647, (8, 432, 21, 21), (952560, 441, 21, 1), 762048)  # alias
        # Source Nodes: [x_669, x_679, x_689, x_comb_iter_1_right_9, x_comb_iter_3_right_9, x_comb_iter_49, x_comb_iter_4_left_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_60.run(buf587, buf642, arg1205_1, arg1206_1, arg559_1, arg560_1, buf588, buf589, buf602, buf610, buf618, buf646, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1205_1
        del arg1206_1
        del arg559_1
        del arg560_1
        del buf587
        del buf642
        # Source Nodes: [x_719, x_720], Original ATen: [aten.convolution, aten.relu]
        buf591 = extern_kernels.convolution(buf590, arg561_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (8, 864, 21, 21), (381024, 441, 21, 1))
        del arg561_1
        buf592 = buf591; del buf591  # reuse
        # Source Nodes: [x_left_9], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_63.run(buf592, arg1208_1, arg1209_1, arg562_1, arg563_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg1208_1
        del arg1209_1
        del arg562_1
        del arg563_1
        buf659 = empty_strided((8, 864, 25, 25), (540000, 1, 21600, 864), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_725, x_727], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_64.run(buf592, buf659, 6912, 625, grid=grid(6912, 625), stream=stream0)
        # Source Nodes: [x_725, x_727, x_728], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf660 = extern_kernels.convolution(buf659, arg20_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf660, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg20_1
        buf661 = empty_strided((8, 864, 11, 11), (104544, 1, 9504, 864), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_730], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf660, buf661, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf660
        # Source Nodes: [x_730], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, arg567_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg567_1
        buf663 = buf661; del buf661  # reuse
        # Source Nodes: [x_731, x_732], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf662, arg1214_1, arg1215_1, arg568_1, arg569_1, buf663, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1214_1
        del arg1215_1
        del arg568_1
        del arg569_1
        del buf662
        # Source Nodes: [x_731, x_732, x_733], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf664 = extern_kernels.convolution(buf663, arg570_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf664, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg570_1
        buf665 = buf663; del buf663  # reuse
        # Source Nodes: [x_735], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf664, buf665, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf664
        # Source Nodes: [x_735], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, arg571_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg571_1
        del buf665
        buf714 = empty((8, 4320, 11, 11), device='cuda', dtype=torch.float32)
        buf711 = reinterpret_tensor(buf714, (8, 864, 11, 11), (522720, 121, 11, 1), 0)  # alias
        # Source Nodes: [x_738, x_comb_iter_0_left_10, x_comb_iter_0_right_10, x_comb_iter_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_67.run(buf592, buf666, arg1217_1, arg1218_1, arg572_1, arg573_1, buf711, 836352, grid=grid(836352), stream=stream0)
        del arg1217_1
        del arg1218_1
        del arg572_1
        del arg573_1
        # Source Nodes: [x_669, x_670], Original ATen: [aten.convolution, aten.relu]
        buf603 = extern_kernels.convolution(buf602, arg521_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf603, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg521_1
        buf604 = buf602; del buf602  # reuse
        # Source Nodes: [x_672], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf603, buf604, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf603
        # Source Nodes: [x_672], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf604, arg522_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg522_1
        buf606 = buf604; del buf604  # reuse
        # Source Nodes: [x_673, x_674], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf605, arg1178_1, arg1179_1, arg523_1, arg524_1, buf606, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1178_1
        del arg1179_1
        del arg523_1
        del arg524_1
        del buf605
        # Source Nodes: [x_673, x_674, x_675], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf607 = extern_kernels.convolution(buf606, arg525_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf607, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg525_1
        buf608 = buf606; del buf606  # reuse
        # Source Nodes: [x_677], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf607, buf608, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf607
        # Source Nodes: [x_677], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf608, arg526_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg526_1
        del buf608
        # Source Nodes: [x_679, x_680], Original ATen: [aten.convolution, aten.relu]
        buf611 = extern_kernels.convolution(buf610, arg529_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf611, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg529_1
        buf612 = buf610; del buf610  # reuse
        # Source Nodes: [x_682], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf611, buf612, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf611
        # Source Nodes: [x_682], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, arg530_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg530_1
        buf614 = buf612; del buf612  # reuse
        # Source Nodes: [x_683, x_684], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf613, arg1184_1, arg1185_1, arg531_1, arg532_1, buf614, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1184_1
        del arg1185_1
        del arg531_1
        del arg532_1
        del buf613
        # Source Nodes: [x_683, x_684, x_685], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf615 = extern_kernels.convolution(buf614, arg533_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf615, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg533_1
        buf616 = buf614; del buf614  # reuse
        # Source Nodes: [x_687], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf615, buf616, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf615
        # Source Nodes: [x_687], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, arg534_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg534_1
        del buf616
        # Source Nodes: [x_689, x_690], Original ATen: [aten.convolution, aten.relu]
        buf619 = extern_kernels.convolution(buf618, arg537_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf619, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg537_1
        buf620 = buf618; del buf618  # reuse
        # Source Nodes: [x_692], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf619, buf620, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf619
        # Source Nodes: [x_692], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf620, arg538_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg538_1
        buf622 = buf620; del buf620  # reuse
        # Source Nodes: [x_693, x_694], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf621, arg1190_1, arg1191_1, arg539_1, arg540_1, buf622, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1190_1
        del arg1191_1
        del arg539_1
        del arg540_1
        del buf621
        # Source Nodes: [x_693, x_694, x_695], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf623 = extern_kernels.convolution(buf622, arg541_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf623, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg541_1
        buf624 = buf622; del buf622  # reuse
        # Source Nodes: [x_697], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf623, buf624, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf623
        # Source Nodes: [x_697], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, arg542_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf625, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg542_1
        buf626 = reinterpret_tensor(buf647, (8, 432, 21, 21), (952560, 441, 21, 1), 381024)  # alias
        buf627 = buf624; del buf624  # reuse
        # Source Nodes: [x_699, x_comb_iter_2_left_9, x_comb_iter_2_right_9, x_comb_iter_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_51.run(buf617, arg1187_1, arg1188_1, arg535_1, arg536_1, buf625, arg1193_1, arg1194_1, arg543_1, arg544_1, buf626, buf627, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1187_1
        del arg1188_1
        del arg1193_1
        del arg1194_1
        del arg535_1
        del arg536_1
        del arg543_1
        del arg544_1
        del buf617
        del buf625
        # Source Nodes: [x_699, x_700], Original ATen: [aten.convolution, aten.relu]
        buf628 = extern_kernels.convolution(buf627, arg545_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf628, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg545_1
        buf629 = buf627; del buf627  # reuse
        # Source Nodes: [x_702], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf628, buf629, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf628
        # Source Nodes: [x_702], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, arg546_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg546_1
        buf631 = buf629; del buf629  # reuse
        # Source Nodes: [x_703, x_704], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf630, arg1196_1, arg1197_1, arg547_1, arg548_1, buf631, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg1196_1
        del arg1197_1
        del arg547_1
        del arg548_1
        del buf630
        # Source Nodes: [x_703, x_704, x_705], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf632 = extern_kernels.convolution(buf631, arg549_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf632, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg549_1
        buf633 = buf631; del buf631  # reuse
        # Source Nodes: [x_707], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf632, buf633, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del buf632
        # Source Nodes: [x_707], Original ATen: [aten.convolution]
        buf634 = extern_kernels.convolution(buf633, arg550_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf634, (8, 432, 21, 21), (190512, 441, 21, 1))
        del arg550_1
        del buf633
        buf644 = reinterpret_tensor(buf647, (8, 432, 21, 21), (952560, 441, 21, 1), 190512)  # alias
        # Source Nodes: [x_comb_iter_1_left_9, x_comb_iter_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf609, arg1181_1, arg1182_1, arg527_1, arg528_1, buf588, buf644, 1524096, grid=grid(1524096), stream=stream0)
        del arg1181_1
        del arg1182_1
        del arg527_1
        del arg528_1
        del buf588
        del buf609
        buf645 = reinterpret_tensor(buf647, (8, 432, 21, 21), (952560, 441, 21, 1), 571536)  # alias
        # Source Nodes: [x_comb_iter_3_left_9, x_comb_iter_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_62.run(buf634, arg1199_1, arg1200_1, arg551_1, arg552_1, buf589, buf645, 1524096, grid=grid(1524096), stream=stream0)
        del arg1199_1
        del arg1200_1
        del arg551_1
        del arg552_1
        del buf589
        del buf634
        buf648 = buf590; del buf590  # reuse
        # Source Nodes: [x_722], Original ATen: [aten.relu]
        triton_poi_fused_relu_68.run(buf647, buf648, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del buf626
        del buf643
        del buf644
        del buf645
        del buf646
        # Source Nodes: [x_722, x_723], Original ATen: [aten.convolution, aten.relu]
        buf649 = extern_kernels.convolution(buf648, arg564_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (8, 864, 21, 21), (381024, 441, 21, 1))
        del arg564_1
        del buf648
        buf650 = buf649; del buf649  # reuse
        buf708 = reinterpret_tensor(buf378, (8, 864, 21, 21), (381024, 1, 18144, 864), 0); del buf378  # reuse
        # Source Nodes: [x_801, x_right_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_69.run(buf650, arg1211_1, arg1212_1, arg565_1, arg566_1, buf708, 6912, 441, grid=grid(6912, 441), stream=stream0)
        del arg1211_1
        del arg1212_1
        del arg565_1
        del arg566_1
        buf667 = empty_strided((8, 864, 27, 27), (629856, 1, 23328, 864), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_739, x_741], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_70.run(buf650, buf667, 6912, 729, grid=grid(6912, 729), stream=stream0)
        # Source Nodes: [x_739, x_741, x_742], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf668 = extern_kernels.convolution(buf667, arg21_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf668, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg21_1
        del buf667
        buf669 = reinterpret_tensor(buf666, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf666  # reuse
        # Source Nodes: [x_744], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf668, buf669, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf668
        # Source Nodes: [x_744], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf669, arg574_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg574_1
        buf671 = buf669; del buf669  # reuse
        # Source Nodes: [x_745, x_746], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf670, arg1220_1, arg1221_1, arg575_1, arg576_1, buf671, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1220_1
        del arg1221_1
        del arg575_1
        del arg576_1
        del buf670
        # Source Nodes: [x_745, x_746, x_747], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf672 = extern_kernels.convolution(buf671, arg577_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf672, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg577_1
        buf673 = buf671; del buf671  # reuse
        # Source Nodes: [x_749], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf672, buf673, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf672
        # Source Nodes: [x_749], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, arg578_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg578_1
        buf675 = buf659; del buf659  # reuse
        # Source Nodes: [x_753, x_755], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_64.run(buf650, buf675, 6912, 625, grid=grid(6912, 625), stream=stream0)
        # Source Nodes: [x_753, x_755, x_756], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf676 = extern_kernels.convolution(buf675, arg22_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf676, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg22_1
        del buf675
        buf677 = buf673; del buf673  # reuse
        # Source Nodes: [x_758], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf676, buf677, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf676
        # Source Nodes: [x_758], Original ATen: [aten.convolution]
        buf678 = extern_kernels.convolution(buf677, arg581_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf678, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg581_1
        buf679 = buf677; del buf677  # reuse
        # Source Nodes: [x_759, x_760], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf678, arg1226_1, arg1227_1, arg582_1, arg583_1, buf679, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1226_1
        del arg1227_1
        del arg582_1
        del arg583_1
        del buf678
        # Source Nodes: [x_759, x_760, x_761], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf680 = extern_kernels.convolution(buf679, arg584_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf680, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg584_1
        buf681 = buf679; del buf679  # reuse
        # Source Nodes: [x_763], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf680, buf681, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf680
        # Source Nodes: [x_763], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf681, arg585_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg585_1
        buf683 = empty_strided((8, 864, 23, 23), (457056, 1, 19872, 864), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_765, x_767], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_71.run(buf650, buf683, 6912, 529, grid=grid(6912, 529), stream=stream0)
        # Source Nodes: [x_765, x_767, x_768], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf684 = extern_kernels.convolution(buf683, arg23_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf684, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg23_1
        buf685 = buf681; del buf681  # reuse
        # Source Nodes: [x_770], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf684, buf685, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf684
        # Source Nodes: [x_770], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, arg588_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg588_1
        buf687 = buf685; del buf685  # reuse
        # Source Nodes: [x_771, x_772], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf686, arg1232_1, arg1233_1, arg589_1, arg590_1, buf687, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1232_1
        del arg1233_1
        del arg589_1
        del arg590_1
        del buf686
        # Source Nodes: [x_771, x_772, x_773], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf688 = extern_kernels.convolution(buf687, arg591_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf688, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg591_1
        buf689 = buf687; del buf687  # reuse
        # Source Nodes: [x_775], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf688, buf689, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf688
        # Source Nodes: [x_775], Original ATen: [aten.convolution]
        buf690 = extern_kernels.convolution(buf689, arg592_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf690, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg592_1
        buf691 = reinterpret_tensor(buf714, (8, 864, 11, 11), (522720, 121, 11, 1), 209088)  # alias
        buf692 = buf689; del buf689  # reuse
        # Source Nodes: [x_777, x_comb_iter_2_left_10, x_comb_iter_2_right_10, x_comb_iter_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72.run(buf682, arg1229_1, arg1230_1, arg586_1, arg587_1, buf690, arg1235_1, arg1236_1, arg593_1, arg594_1, buf691, buf692, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1229_1
        del arg1230_1
        del arg1235_1
        del arg1236_1
        del arg586_1
        del arg587_1
        del arg593_1
        del arg594_1
        del buf682
        del buf690
        # Source Nodes: [x_777, x_778], Original ATen: [aten.convolution, aten.relu]
        buf693 = extern_kernels.convolution(buf692, arg595_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf693, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg595_1
        buf694 = buf692; del buf692  # reuse
        # Source Nodes: [x_780], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf693, buf694, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf693
        # Source Nodes: [x_780], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, arg596_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg596_1
        buf696 = buf694; del buf694  # reuse
        # Source Nodes: [x_781, x_782], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf695, arg1238_1, arg1239_1, arg597_1, arg598_1, buf696, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1238_1
        del arg1239_1
        del arg597_1
        del arg598_1
        del buf695
        # Source Nodes: [x_781, x_782, x_783], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf697 = extern_kernels.convolution(buf696, arg599_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf697, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg599_1
        buf698 = buf696; del buf696  # reuse
        # Source Nodes: [x_785], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf697, buf698, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf697
        # Source Nodes: [x_785], Original ATen: [aten.convolution]
        buf699 = extern_kernels.convolution(buf698, arg600_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf699, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg600_1
        buf712 = reinterpret_tensor(buf714, (8, 864, 11, 11), (522720, 121, 11, 1), 104544)  # alias
        buf713 = reinterpret_tensor(buf714, (8, 864, 11, 11), (522720, 121, 11, 1), 313632)  # alias
        # Source Nodes: [x_752, x_788, x_comb_iter_1_left_10, x_comb_iter_1_right_10, x_comb_iter_3_left_10, x_comb_iter_3_right_10, x_comb_iter_51, x_comb_iter_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_73.run(buf650, buf674, arg1223_1, arg1224_1, arg579_1, arg580_1, buf699, arg1241_1, arg1242_1, arg601_1, arg602_1, buf712, buf713, 836352, grid=grid(836352), stream=stream0)
        del arg1223_1
        del arg1224_1
        del arg1241_1
        del arg1242_1
        del arg579_1
        del arg580_1
        del arg601_1
        del arg602_1
        del buf650
        buf653 = empty_strided((8, 2160, 11, 11), (261360, 1, 23760, 2160), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_1_avgpool, x_806], Original ATen: [aten.avg_pool2d, aten.relu]
        triton_poi_fused_avg_pool2d_relu_74.run(buf647, buf653, 17280, 121, grid=grid(17280, 121), stream=stream0)
        # Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_1_avgpool, x_806, x_path1_3], Original ATen: [aten.avg_pool2d, aten.convolution, aten.relu]
        buf654 = extern_kernels.convolution(buf653, arg612_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (8, 432, 11, 11), (52272, 121, 11, 1))
        del arg612_1
        buf655 = buf653; del buf653  # reuse
        # Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_2_avgpool, l__mod___cell_9_conv_prev_1x1_path_2_pad, x_806], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.relu]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_75.run(buf647, buf655, 17280, 121, grid=grid(17280, 121), stream=stream0)
        del buf647
        # Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_2_avgpool, l__mod___cell_9_conv_prev_1x1_path_2_pad, x_806, x_path2_3], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.relu]
        buf656 = extern_kernels.convolution(buf655, arg613_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (8, 432, 11, 11), (52272, 121, 11, 1))
        del arg613_1
        del buf655
        buf657 = buf699; del buf699  # reuse
        buf724 = reinterpret_tensor(buf674, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf674  # reuse
        buf765 = buf698; del buf698  # reuse
        # Source Nodes: [cat_21, x_810, x_860, x_left_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76.run(buf654, buf656, arg1253_1, arg1254_1, arg614_1, arg615_1, buf657, buf724, buf765, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1253_1
        del arg1254_1
        del arg614_1
        del arg615_1
        del buf654
        del buf656
        # Source Nodes: [x_810, x_811], Original ATen: [aten.convolution, aten.relu]
        buf725 = extern_kernels.convolution(buf724, arg619_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf725, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg619_1
        buf726 = buf724; del buf724  # reuse
        # Source Nodes: [x_813], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf725, buf726, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf725
        # Source Nodes: [x_813], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf726, arg620_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf727, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg620_1
        buf728 = buf726; del buf726  # reuse
        # Source Nodes: [x_814, x_815], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf727, arg1259_1, arg1260_1, arg621_1, arg622_1, buf728, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1259_1
        del arg1260_1
        del arg621_1
        del arg622_1
        del buf727
        # Source Nodes: [x_814, x_815, x_816], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf729 = extern_kernels.convolution(buf728, arg623_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf729, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg623_1
        buf730 = buf728; del buf728  # reuse
        # Source Nodes: [x_818], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf729, buf730, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf729
        # Source Nodes: [x_818], Original ATen: [aten.convolution]
        buf731 = extern_kernels.convolution(buf730, arg624_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf731, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg624_1
        del buf730
        buf777 = empty((8, 4320, 11, 11), device='cuda', dtype=torch.float32)
        buf773 = reinterpret_tensor(buf777, (8, 864, 11, 11), (522720, 121, 11, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_11, x_comb_iter_0_right_11, x_comb_iter_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_77.run(buf657, buf731, arg1262_1, arg1263_1, arg625_1, arg626_1, buf773, 836352, grid=grid(836352), stream=stream0)
        del arg1262_1
        del arg1263_1
        del arg625_1
        del arg626_1
        del buf657
        buf700 = buf683; del buf683  # reuse
        # Source Nodes: [x_789, x_791], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_71.run(buf592, buf700, 6912, 529, grid=grid(6912, 529), stream=stream0)
        del buf592
        # Source Nodes: [x_789, x_791, x_792], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.relu]
        buf701 = extern_kernels.convolution(buf700, arg24_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf701, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg24_1
        del buf700
        buf702 = reinterpret_tensor(buf731, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf731  # reuse
        # Source Nodes: [x_794], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf701, buf702, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf701
        # Source Nodes: [x_794], Original ATen: [aten.convolution]
        buf703 = extern_kernels.convolution(buf702, arg603_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg603_1
        buf704 = buf702; del buf702  # reuse
        # Source Nodes: [x_795, x_796], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf703, arg1244_1, arg1245_1, arg604_1, arg605_1, buf704, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1244_1
        del arg1245_1
        del arg604_1
        del arg605_1
        del buf703
        # Source Nodes: [x_795, x_796, x_797], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf705 = extern_kernels.convolution(buf704, arg606_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf705, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg606_1
        buf706 = buf704; del buf704  # reuse
        # Source Nodes: [x_799], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf705, buf706, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf705
        # Source Nodes: [x_799], Original ATen: [aten.convolution]
        buf707 = extern_kernels.convolution(buf706, arg607_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf707, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg607_1
        del buf706
        # Source Nodes: [x_801, x_804], Original ATen: [aten.convolution, aten.relu]
        buf709 = extern_kernels.convolution(buf708, arg25_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg25_1
        del buf708
        buf710 = reinterpret_tensor(buf714, (8, 864, 11, 11), (522720, 121, 11, 1), 418176)  # alias
        # Source Nodes: [x_comb_iter_4_left_10, x_comb_iter_4_right_10, x_comb_iter_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_78.run(buf707, arg1247_1, arg1248_1, arg608_1, arg609_1, buf709, arg1250_1, arg1251_1, arg610_1, arg611_1, buf710, 836352, grid=grid(836352), stream=stream0)
        del arg1247_1
        del arg1248_1
        del arg1250_1
        del arg1251_1
        del arg608_1
        del arg609_1
        del arg610_1
        del arg611_1
        del buf707
        buf715 = empty_strided((8, 4320, 11, 11), (522720, 1, 47520, 4320), device='cuda', dtype=torch.float32)
        buf720 = empty_strided((8, 4320, 11, 11), (522720, 1, 47520, 4320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_807, x_870], Original ATen: [aten.relu]
        triton_poi_fused_relu_79.run(buf714, buf715, buf720, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del buf691
        del buf710
        del buf711
        del buf712
        del buf713
        # Source Nodes: [x_807, x_808], Original ATen: [aten.convolution, aten.relu]
        buf716 = extern_kernels.convolution(buf715, arg616_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf716, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg616_1
        buf717 = buf716; del buf716  # reuse
        # Source Nodes: [x_comb_iter_4_right_11], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_80.run(buf717, arg1256_1, arg1257_1, arg617_1, arg618_1, 836352, grid=grid(836352), stream=stream0)
        del arg1256_1
        del arg1257_1
        del arg617_1
        del arg618_1
        # Source Nodes: [x_860, x_861], Original ATen: [aten.convolution, aten.relu]
        buf766 = extern_kernels.convolution(buf765, arg659_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf766, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg659_1
        buf767 = buf765; del buf765  # reuse
        # Source Nodes: [x_863], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf766, buf767, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_863], Original ATen: [aten.convolution]
        buf768 = extern_kernels.convolution(buf767, arg660_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf768, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg660_1
        buf769 = buf767; del buf767  # reuse
        # Source Nodes: [x_864, x_865], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf768, arg1289_1, arg1290_1, arg661_1, arg662_1, buf769, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1289_1
        del arg1290_1
        del arg661_1
        del arg662_1
        # Source Nodes: [x_864, x_865, x_866], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf770 = extern_kernels.convolution(buf769, arg663_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf770, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg663_1
        buf771 = buf769; del buf769  # reuse
        # Source Nodes: [x_868], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf770, buf771, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_868], Original ATen: [aten.convolution]
        buf772 = extern_kernels.convolution(buf771, arg664_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf772, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg664_1
        buf718 = reinterpret_tensor(buf771, (8, 864, 11, 11), (104544, 121, 11, 1), 0); del buf771  # reuse
        buf719 = buf770; del buf770  # reuse
        buf732 = reinterpret_tensor(buf768, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf768  # reuse
        buf740 = reinterpret_tensor(buf766, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf766  # reuse
        buf748 = reinterpret_tensor(buf709, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf709  # reuse
        buf776 = reinterpret_tensor(buf777, (8, 864, 11, 11), (522720, 121, 11, 1), 418176)  # alias
        # Source Nodes: [x_820, x_830, x_840, x_comb_iter_1_right_11, x_comb_iter_3_right_11, x_comb_iter_4_left_11, x_comb_iter_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_81.run(buf717, buf772, arg1292_1, arg1293_1, arg665_1, arg666_1, buf718, buf719, buf732, buf740, buf748, buf776, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1292_1
        del arg1293_1
        del arg665_1
        del arg666_1
        # Source Nodes: [x_870, x_871], Original ATen: [aten.convolution, aten.relu]
        buf721 = extern_kernels.convolution(buf720, arg667_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf721, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg667_1
        buf722 = buf721; del buf721  # reuse
        buf787 = reinterpret_tensor(buf772, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf772  # reuse
        buf828 = reinterpret_tensor(buf717, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf717  # reuse
        # Source Nodes: [x_876, x_926, x_left_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_82.run(buf722, arg1295_1, arg1296_1, arg668_1, arg669_1, buf787, buf828, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1295_1
        del arg1296_1
        del arg668_1
        del arg669_1
        # Source Nodes: [x_876, x_877], Original ATen: [aten.convolution, aten.relu]
        buf788 = extern_kernels.convolution(buf787, arg673_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf788, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg673_1
        buf789 = buf787; del buf787  # reuse
        # Source Nodes: [x_879], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf788, buf789, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf788
        # Source Nodes: [x_879], Original ATen: [aten.convolution]
        buf790 = extern_kernels.convolution(buf789, arg674_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf790, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg674_1
        buf791 = buf789; del buf789  # reuse
        # Source Nodes: [x_880, x_881], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf790, arg1301_1, arg1302_1, arg675_1, arg676_1, buf791, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1301_1
        del arg1302_1
        del arg675_1
        del arg676_1
        del buf790
        # Source Nodes: [x_880, x_881, x_882], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf792 = extern_kernels.convolution(buf791, arg677_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf792, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg677_1
        buf793 = buf791; del buf791  # reuse
        # Source Nodes: [x_884], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf792, buf793, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf792
        # Source Nodes: [x_884], Original ATen: [aten.convolution]
        buf794 = extern_kernels.convolution(buf793, arg678_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf794, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg678_1
        del buf793
        buf840 = reinterpret_tensor(buf720, (8, 4320, 11, 11), (522720, 121, 11, 1), 0); del buf720  # reuse
        buf836 = reinterpret_tensor(buf840, (8, 864, 11, 11), (522720, 121, 11, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_12, x_comb_iter_0_right_12, x_comb_iter_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_77.run(buf722, buf794, arg1304_1, arg1305_1, arg679_1, arg680_1, buf836, 836352, grid=grid(836352), stream=stream0)
        del arg1304_1
        del arg1305_1
        del arg679_1
        del arg680_1
        del buf722
        del buf794
        # Source Nodes: [x_820, x_821], Original ATen: [aten.convolution, aten.relu]
        buf733 = extern_kernels.convolution(buf732, arg627_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf733, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg627_1
        buf734 = buf732; del buf732  # reuse
        # Source Nodes: [x_823], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf733, buf734, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf733
        # Source Nodes: [x_823], Original ATen: [aten.convolution]
        buf735 = extern_kernels.convolution(buf734, arg628_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf735, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg628_1
        buf736 = buf734; del buf734  # reuse
        # Source Nodes: [x_824, x_825], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf735, arg1265_1, arg1266_1, arg629_1, arg630_1, buf736, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1265_1
        del arg1266_1
        del arg629_1
        del arg630_1
        del buf735
        # Source Nodes: [x_824, x_825, x_826], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf737 = extern_kernels.convolution(buf736, arg631_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf737, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg631_1
        buf738 = buf736; del buf736  # reuse
        # Source Nodes: [x_828], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf737, buf738, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf737
        # Source Nodes: [x_828], Original ATen: [aten.convolution]
        buf739 = extern_kernels.convolution(buf738, arg632_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf739, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg632_1
        del buf738
        # Source Nodes: [x_830, x_831], Original ATen: [aten.convolution, aten.relu]
        buf741 = extern_kernels.convolution(buf740, arg635_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf741, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg635_1
        buf742 = buf740; del buf740  # reuse
        # Source Nodes: [x_833], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf741, buf742, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf741
        # Source Nodes: [x_833], Original ATen: [aten.convolution]
        buf743 = extern_kernels.convolution(buf742, arg636_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf743, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg636_1
        buf744 = buf742; del buf742  # reuse
        # Source Nodes: [x_834, x_835], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf743, arg1271_1, arg1272_1, arg637_1, arg638_1, buf744, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1271_1
        del arg1272_1
        del arg637_1
        del arg638_1
        del buf743
        # Source Nodes: [x_834, x_835, x_836], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf745 = extern_kernels.convolution(buf744, arg639_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf745, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg639_1
        buf746 = buf744; del buf744  # reuse
        # Source Nodes: [x_838], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf745, buf746, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf745
        # Source Nodes: [x_838], Original ATen: [aten.convolution]
        buf747 = extern_kernels.convolution(buf746, arg640_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf747, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg640_1
        del buf746
        # Source Nodes: [x_840, x_841], Original ATen: [aten.convolution, aten.relu]
        buf749 = extern_kernels.convolution(buf748, arg643_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf749, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg643_1
        buf750 = buf748; del buf748  # reuse
        # Source Nodes: [x_843], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf749, buf750, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf749
        # Source Nodes: [x_843], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, arg644_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg644_1
        buf752 = buf750; del buf750  # reuse
        # Source Nodes: [x_844, x_845], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf751, arg1277_1, arg1278_1, arg645_1, arg646_1, buf752, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1277_1
        del arg1278_1
        del arg645_1
        del arg646_1
        del buf751
        # Source Nodes: [x_844, x_845, x_846], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf753 = extern_kernels.convolution(buf752, arg647_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf753, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg647_1
        buf754 = buf752; del buf752  # reuse
        # Source Nodes: [x_848], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf753, buf754, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf753
        # Source Nodes: [x_848], Original ATen: [aten.convolution]
        buf755 = extern_kernels.convolution(buf754, arg648_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf755, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg648_1
        buf756 = reinterpret_tensor(buf777, (8, 864, 11, 11), (522720, 121, 11, 1), 209088)  # alias
        buf757 = buf754; del buf754  # reuse
        # Source Nodes: [x_850, x_comb_iter_2_left_11, x_comb_iter_2_right_11, x_comb_iter_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72.run(buf747, arg1274_1, arg1275_1, arg641_1, arg642_1, buf755, arg1280_1, arg1281_1, arg649_1, arg650_1, buf756, buf757, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1274_1
        del arg1275_1
        del arg1280_1
        del arg1281_1
        del arg641_1
        del arg642_1
        del arg649_1
        del arg650_1
        del buf747
        del buf755
        # Source Nodes: [x_850, x_851], Original ATen: [aten.convolution, aten.relu]
        buf758 = extern_kernels.convolution(buf757, arg651_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf758, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg651_1
        buf759 = buf757; del buf757  # reuse
        # Source Nodes: [x_853], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf758, buf759, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf758
        # Source Nodes: [x_853], Original ATen: [aten.convolution]
        buf760 = extern_kernels.convolution(buf759, arg652_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf760, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg652_1
        buf761 = buf759; del buf759  # reuse
        # Source Nodes: [x_854, x_855], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf760, arg1283_1, arg1284_1, arg653_1, arg654_1, buf761, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1283_1
        del arg1284_1
        del arg653_1
        del arg654_1
        del buf760
        # Source Nodes: [x_854, x_855, x_856], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf762 = extern_kernels.convolution(buf761, arg655_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf762, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg655_1
        buf763 = buf761; del buf761  # reuse
        # Source Nodes: [x_858], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf762, buf763, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf762
        # Source Nodes: [x_858], Original ATen: [aten.convolution]
        buf764 = extern_kernels.convolution(buf763, arg656_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf764, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg656_1
        del buf763
        buf774 = reinterpret_tensor(buf777, (8, 864, 11, 11), (522720, 121, 11, 1), 104544)  # alias
        # Source Nodes: [x_comb_iter_1_left_11, x_comb_iter_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf739, arg1268_1, arg1269_1, arg633_1, arg634_1, buf718, buf774, 836352, grid=grid(836352), stream=stream0)
        del arg1268_1
        del arg1269_1
        del arg633_1
        del arg634_1
        del buf718
        del buf739
        buf775 = reinterpret_tensor(buf777, (8, 864, 11, 11), (522720, 121, 11, 1), 313632)  # alias
        # Source Nodes: [x_comb_iter_3_left_11, x_comb_iter_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf764, arg1286_1, arg1287_1, arg657_1, arg658_1, buf719, buf775, 836352, grid=grid(836352), stream=stream0)
        del arg1286_1
        del arg1287_1
        del arg657_1
        del arg658_1
        del buf719
        buf778 = buf715; del buf715  # reuse
        buf783 = reinterpret_tensor(buf714, (8, 4320, 11, 11), (522720, 1, 47520, 4320), 0); del buf714  # reuse
        # Source Nodes: [x_873, x_936], Original ATen: [aten.relu]
        triton_poi_fused_relu_79.run(buf777, buf778, buf783, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del buf756
        del buf773
        del buf774
        del buf775
        del buf776
        del buf777
        # Source Nodes: [x_873, x_874], Original ATen: [aten.convolution, aten.relu]
        buf779 = extern_kernels.convolution(buf778, arg670_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf779, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg670_1
        buf780 = buf779; del buf779  # reuse
        # Source Nodes: [x_comb_iter_4_right_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_80.run(buf780, arg1298_1, arg1299_1, arg671_1, arg672_1, 836352, grid=grid(836352), stream=stream0)
        del arg1298_1
        del arg1299_1
        del arg671_1
        del arg672_1
        # Source Nodes: [x_926, x_927], Original ATen: [aten.convolution, aten.relu]
        buf829 = extern_kernels.convolution(buf828, arg713_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf829, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg713_1
        buf830 = buf828; del buf828  # reuse
        # Source Nodes: [x_929], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf829, buf830, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_929], Original ATen: [aten.convolution]
        buf831 = extern_kernels.convolution(buf830, arg714_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf831, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg714_1
        buf832 = buf830; del buf830  # reuse
        # Source Nodes: [x_930, x_931], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf831, arg1331_1, arg1332_1, arg715_1, arg716_1, buf832, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1331_1
        del arg1332_1
        del arg715_1
        del arg716_1
        # Source Nodes: [x_930, x_931, x_932], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf833 = extern_kernels.convolution(buf832, arg717_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf833, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg717_1
        buf834 = buf832; del buf832  # reuse
        # Source Nodes: [x_934], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf833, buf834, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_934], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf834, arg718_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg718_1
        buf781 = reinterpret_tensor(buf834, (8, 864, 11, 11), (104544, 121, 11, 1), 0); del buf834  # reuse
        buf782 = buf833; del buf833  # reuse
        buf795 = reinterpret_tensor(buf831, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf831  # reuse
        buf803 = reinterpret_tensor(buf829, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf829  # reuse
        buf811 = reinterpret_tensor(buf764, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf764  # reuse
        buf839 = reinterpret_tensor(buf840, (8, 864, 11, 11), (522720, 121, 11, 1), 418176)  # alias
        # Source Nodes: [x_886, x_896, x_906, x_comb_iter_1_right_12, x_comb_iter_3_right_12, x_comb_iter_4_left_12, x_comb_iter_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_81.run(buf780, buf835, arg1334_1, arg1335_1, arg719_1, arg720_1, buf781, buf782, buf795, buf803, buf811, buf839, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1334_1
        del arg1335_1
        del arg719_1
        del arg720_1
        # Source Nodes: [x_936, x_937], Original ATen: [aten.convolution, aten.relu]
        buf784 = extern_kernels.convolution(buf783, arg721_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf784, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg721_1
        buf785 = buf784; del buf784  # reuse
        buf846 = reinterpret_tensor(buf835, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf835  # reuse
        buf887 = reinterpret_tensor(buf780, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf780  # reuse
        # Source Nodes: [x_942, x_992, x_left_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_82.run(buf785, arg1337_1, arg1338_1, arg722_1, arg723_1, buf846, buf887, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1337_1
        del arg1338_1
        del arg722_1
        del arg723_1
        # Source Nodes: [x_942, x_943], Original ATen: [aten.convolution, aten.relu]
        buf847 = extern_kernels.convolution(buf846, arg727_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf847, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg727_1
        buf848 = buf846; del buf846  # reuse
        # Source Nodes: [x_945], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf847, buf848, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf847
        # Source Nodes: [x_945], Original ATen: [aten.convolution]
        buf849 = extern_kernels.convolution(buf848, arg728_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf849, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg728_1
        buf850 = buf848; del buf848  # reuse
        # Source Nodes: [x_946, x_947], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf849, arg1343_1, arg1344_1, arg729_1, arg730_1, buf850, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1343_1
        del arg1344_1
        del arg729_1
        del arg730_1
        del buf849
        # Source Nodes: [x_946, x_947, x_948], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf851 = extern_kernels.convolution(buf850, arg731_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf851, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg731_1
        buf852 = buf850; del buf850  # reuse
        # Source Nodes: [x_950], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf851, buf852, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf851
        # Source Nodes: [x_950], Original ATen: [aten.convolution]
        buf853 = extern_kernels.convolution(buf852, arg732_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf853, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg732_1
        del buf852
        buf899 = reinterpret_tensor(buf783, (8, 4320, 11, 11), (522720, 121, 11, 1), 0); del buf783  # reuse
        buf895 = reinterpret_tensor(buf899, (8, 864, 11, 11), (522720, 121, 11, 1), 0)  # alias
        # Source Nodes: [x_comb_iter_0_left_13, x_comb_iter_0_right_13, x_comb_iter_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_77.run(buf785, buf853, arg1346_1, arg1347_1, arg733_1, arg734_1, buf895, 836352, grid=grid(836352), stream=stream0)
        del arg1346_1
        del arg1347_1
        del arg733_1
        del arg734_1
        del buf785
        del buf853
        # Source Nodes: [x_886, x_887], Original ATen: [aten.convolution, aten.relu]
        buf796 = extern_kernels.convolution(buf795, arg681_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf796, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg681_1
        buf797 = buf795; del buf795  # reuse
        # Source Nodes: [x_889], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf796, buf797, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf796
        # Source Nodes: [x_889], Original ATen: [aten.convolution]
        buf798 = extern_kernels.convolution(buf797, arg682_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf798, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg682_1
        buf799 = buf797; del buf797  # reuse
        # Source Nodes: [x_890, x_891], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf798, arg1307_1, arg1308_1, arg683_1, arg684_1, buf799, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1307_1
        del arg1308_1
        del arg683_1
        del arg684_1
        del buf798
        # Source Nodes: [x_890, x_891, x_892], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf800 = extern_kernels.convolution(buf799, arg685_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf800, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg685_1
        buf801 = buf799; del buf799  # reuse
        # Source Nodes: [x_894], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf800, buf801, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf800
        # Source Nodes: [x_894], Original ATen: [aten.convolution]
        buf802 = extern_kernels.convolution(buf801, arg686_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf802, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg686_1
        del buf801
        # Source Nodes: [x_896, x_897], Original ATen: [aten.convolution, aten.relu]
        buf804 = extern_kernels.convolution(buf803, arg689_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf804, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg689_1
        buf805 = buf803; del buf803  # reuse
        # Source Nodes: [x_899], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf804, buf805, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf804
        # Source Nodes: [x_899], Original ATen: [aten.convolution]
        buf806 = extern_kernels.convolution(buf805, arg690_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf806, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg690_1
        buf807 = buf805; del buf805  # reuse
        # Source Nodes: [x_900, x_901], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf806, arg1313_1, arg1314_1, arg691_1, arg692_1, buf807, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1313_1
        del arg1314_1
        del arg691_1
        del arg692_1
        del buf806
        # Source Nodes: [x_900, x_901, x_902], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf808 = extern_kernels.convolution(buf807, arg693_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf808, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg693_1
        buf809 = buf807; del buf807  # reuse
        # Source Nodes: [x_904], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf808, buf809, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf808
        # Source Nodes: [x_904], Original ATen: [aten.convolution]
        buf810 = extern_kernels.convolution(buf809, arg694_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf810, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg694_1
        del buf809
        # Source Nodes: [x_906, x_907], Original ATen: [aten.convolution, aten.relu]
        buf812 = extern_kernels.convolution(buf811, arg697_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf812, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg697_1
        buf813 = buf811; del buf811  # reuse
        # Source Nodes: [x_909], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf812, buf813, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf812
        # Source Nodes: [x_909], Original ATen: [aten.convolution]
        buf814 = extern_kernels.convolution(buf813, arg698_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf814, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg698_1
        buf815 = buf813; del buf813  # reuse
        # Source Nodes: [x_910, x_911], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf814, arg1319_1, arg1320_1, arg699_1, arg700_1, buf815, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1319_1
        del arg1320_1
        del arg699_1
        del arg700_1
        del buf814
        # Source Nodes: [x_910, x_911, x_912], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf816 = extern_kernels.convolution(buf815, arg701_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf816, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg701_1
        buf817 = buf815; del buf815  # reuse
        # Source Nodes: [x_914], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf816, buf817, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf816
        # Source Nodes: [x_914], Original ATen: [aten.convolution]
        buf818 = extern_kernels.convolution(buf817, arg702_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf818, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg702_1
        buf819 = reinterpret_tensor(buf840, (8, 864, 11, 11), (522720, 121, 11, 1), 209088)  # alias
        buf820 = buf817; del buf817  # reuse
        # Source Nodes: [x_916, x_comb_iter_2_left_12, x_comb_iter_2_right_12, x_comb_iter_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72.run(buf810, arg1316_1, arg1317_1, arg695_1, arg696_1, buf818, arg1322_1, arg1323_1, arg703_1, arg704_1, buf819, buf820, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1316_1
        del arg1317_1
        del arg1322_1
        del arg1323_1
        del arg695_1
        del arg696_1
        del arg703_1
        del arg704_1
        del buf810
        del buf818
        # Source Nodes: [x_916, x_917], Original ATen: [aten.convolution, aten.relu]
        buf821 = extern_kernels.convolution(buf820, arg705_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf821, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg705_1
        buf822 = buf820; del buf820  # reuse
        # Source Nodes: [x_919], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf821, buf822, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf821
        # Source Nodes: [x_919], Original ATen: [aten.convolution]
        buf823 = extern_kernels.convolution(buf822, arg706_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf823, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg706_1
        buf824 = buf822; del buf822  # reuse
        # Source Nodes: [x_920, x_921], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf823, arg1325_1, arg1326_1, arg707_1, arg708_1, buf824, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1325_1
        del arg1326_1
        del arg707_1
        del arg708_1
        del buf823
        # Source Nodes: [x_920, x_921, x_922], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf825 = extern_kernels.convolution(buf824, arg709_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf825, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg709_1
        buf826 = buf824; del buf824  # reuse
        # Source Nodes: [x_924], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf825, buf826, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf825
        # Source Nodes: [x_924], Original ATen: [aten.convolution]
        buf827 = extern_kernels.convolution(buf826, arg710_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf827, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg710_1
        del buf826
        buf837 = reinterpret_tensor(buf840, (8, 864, 11, 11), (522720, 121, 11, 1), 104544)  # alias
        # Source Nodes: [x_comb_iter_1_left_12, x_comb_iter_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf802, arg1310_1, arg1311_1, arg687_1, arg688_1, buf781, buf837, 836352, grid=grid(836352), stream=stream0)
        del arg1310_1
        del arg1311_1
        del arg687_1
        del arg688_1
        del buf781
        del buf802
        buf838 = reinterpret_tensor(buf840, (8, 864, 11, 11), (522720, 121, 11, 1), 313632)  # alias
        # Source Nodes: [x_comb_iter_3_left_12, x_comb_iter_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf827, arg1328_1, arg1329_1, arg711_1, arg712_1, buf782, buf838, 836352, grid=grid(836352), stream=stream0)
        del arg1328_1
        del arg1329_1
        del arg711_1
        del arg712_1
        del buf782
        buf841 = buf778; del buf778  # reuse
        # Source Nodes: [x_939], Original ATen: [aten.relu]
        triton_poi_fused_relu_84.run(buf840, buf841, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del buf819
        del buf836
        del buf837
        del buf838
        del buf839
        del buf840
        # Source Nodes: [x_939, x_940], Original ATen: [aten.convolution, aten.relu]
        buf842 = extern_kernels.convolution(buf841, arg724_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf842, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg724_1
        del buf841
        buf843 = buf842; del buf842  # reuse
        # Source Nodes: [x_comb_iter_4_right_13], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_80.run(buf843, arg1340_1, arg1341_1, arg725_1, arg726_1, 836352, grid=grid(836352), stream=stream0)
        del arg1340_1
        del arg1341_1
        del arg725_1
        del arg726_1
        # Source Nodes: [x_992, x_993], Original ATen: [aten.convolution, aten.relu]
        buf888 = extern_kernels.convolution(buf887, arg767_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf888, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg767_1
        buf889 = buf887; del buf887  # reuse
        # Source Nodes: [x_995], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf888, buf889, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_995], Original ATen: [aten.convolution]
        buf890 = extern_kernels.convolution(buf889, arg768_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf890, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg768_1
        buf891 = buf889; del buf889  # reuse
        # Source Nodes: [x_996, x_997], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf890, arg1373_1, arg1374_1, arg769_1, arg770_1, buf891, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1373_1
        del arg1374_1
        del arg769_1
        del arg770_1
        # Source Nodes: [x_996, x_997, x_998], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf892 = extern_kernels.convolution(buf891, arg771_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf892, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg771_1
        buf893 = buf891; del buf891  # reuse
        # Source Nodes: [x_1000], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf892, buf893, 6912, 121, grid=grid(6912, 121), stream=stream0)
        # Source Nodes: [x_1000], Original ATen: [aten.convolution]
        buf894 = extern_kernels.convolution(buf893, arg772_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf894, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg772_1
        buf844 = reinterpret_tensor(buf893, (8, 864, 11, 11), (104544, 121, 11, 1), 0); del buf893  # reuse
        buf845 = buf892; del buf892  # reuse
        buf854 = reinterpret_tensor(buf890, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf890  # reuse
        buf862 = reinterpret_tensor(buf888, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf888  # reuse
        buf870 = reinterpret_tensor(buf827, (8, 864, 11, 11), (104544, 1, 9504, 864), 0); del buf827  # reuse
        buf898 = reinterpret_tensor(buf899, (8, 864, 11, 11), (522720, 121, 11, 1), 418176)  # alias
        # Source Nodes: [x_952, x_962, x_972, x_comb_iter_1_right_13, x_comb_iter_3_right_13, x_comb_iter_4_left_13, x_comb_iter_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_81.run(buf843, buf894, arg1376_1, arg1377_1, arg773_1, arg774_1, buf844, buf845, buf854, buf862, buf870, buf898, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1376_1
        del arg1377_1
        del arg773_1
        del arg774_1
        del buf843
        del buf894
        # Source Nodes: [x_952, x_953], Original ATen: [aten.convolution, aten.relu]
        buf855 = extern_kernels.convolution(buf854, arg735_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf855, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg735_1
        buf856 = buf854; del buf854  # reuse
        # Source Nodes: [x_955], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf855, buf856, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf855
        # Source Nodes: [x_955], Original ATen: [aten.convolution]
        buf857 = extern_kernels.convolution(buf856, arg736_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf857, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg736_1
        buf858 = buf856; del buf856  # reuse
        # Source Nodes: [x_956, x_957], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf857, arg1349_1, arg1350_1, arg737_1, arg738_1, buf858, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1349_1
        del arg1350_1
        del arg737_1
        del arg738_1
        del buf857
        # Source Nodes: [x_956, x_957, x_958], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf859 = extern_kernels.convolution(buf858, arg739_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf859, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg739_1
        buf860 = buf858; del buf858  # reuse
        # Source Nodes: [x_960], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf859, buf860, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf859
        # Source Nodes: [x_960], Original ATen: [aten.convolution]
        buf861 = extern_kernels.convolution(buf860, arg740_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg740_1
        del buf860
        # Source Nodes: [x_962, x_963], Original ATen: [aten.convolution, aten.relu]
        buf863 = extern_kernels.convolution(buf862, arg743_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf863, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg743_1
        buf864 = buf862; del buf862  # reuse
        # Source Nodes: [x_965], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf863, buf864, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf863
        # Source Nodes: [x_965], Original ATen: [aten.convolution]
        buf865 = extern_kernels.convolution(buf864, arg744_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf865, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg744_1
        buf866 = buf864; del buf864  # reuse
        # Source Nodes: [x_966, x_967], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf865, arg1355_1, arg1356_1, arg745_1, arg746_1, buf866, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1355_1
        del arg1356_1
        del arg745_1
        del arg746_1
        del buf865
        # Source Nodes: [x_966, x_967, x_968], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf867 = extern_kernels.convolution(buf866, arg747_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf867, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg747_1
        buf868 = buf866; del buf866  # reuse
        # Source Nodes: [x_970], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf867, buf868, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf867
        # Source Nodes: [x_970], Original ATen: [aten.convolution]
        buf869 = extern_kernels.convolution(buf868, arg748_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf869, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg748_1
        del buf868
        # Source Nodes: [x_972, x_973], Original ATen: [aten.convolution, aten.relu]
        buf871 = extern_kernels.convolution(buf870, arg751_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf871, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg751_1
        buf872 = buf870; del buf870  # reuse
        # Source Nodes: [x_975], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf871, buf872, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf871
        # Source Nodes: [x_975], Original ATen: [aten.convolution]
        buf873 = extern_kernels.convolution(buf872, arg752_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf873, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg752_1
        buf874 = buf872; del buf872  # reuse
        # Source Nodes: [x_976, x_977], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf873, arg1361_1, arg1362_1, arg753_1, arg754_1, buf874, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1361_1
        del arg1362_1
        del arg753_1
        del arg754_1
        del buf873
        # Source Nodes: [x_976, x_977, x_978], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf875 = extern_kernels.convolution(buf874, arg755_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf875, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg755_1
        buf876 = buf874; del buf874  # reuse
        # Source Nodes: [x_980], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf875, buf876, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf875
        # Source Nodes: [x_980], Original ATen: [aten.convolution]
        buf877 = extern_kernels.convolution(buf876, arg756_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf877, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg756_1
        buf878 = reinterpret_tensor(buf899, (8, 864, 11, 11), (522720, 121, 11, 1), 209088)  # alias
        buf879 = buf876; del buf876  # reuse
        # Source Nodes: [x_982, x_comb_iter_2_left_13, x_comb_iter_2_right_13, x_comb_iter_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_72.run(buf869, arg1358_1, arg1359_1, arg749_1, arg750_1, buf877, arg1364_1, arg1365_1, arg757_1, arg758_1, buf878, buf879, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1358_1
        del arg1359_1
        del arg1364_1
        del arg1365_1
        del arg749_1
        del arg750_1
        del arg757_1
        del arg758_1
        del buf869
        del buf877
        # Source Nodes: [x_982, x_983], Original ATen: [aten.convolution, aten.relu]
        buf880 = extern_kernels.convolution(buf879, arg759_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf880, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg759_1
        buf881 = buf879; del buf879  # reuse
        # Source Nodes: [x_985], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf880, buf881, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf880
        # Source Nodes: [x_985], Original ATen: [aten.convolution]
        buf882 = extern_kernels.convolution(buf881, arg760_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf882, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg760_1
        buf883 = buf881; del buf881  # reuse
        # Source Nodes: [x_986, x_987], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf882, arg1367_1, arg1368_1, arg761_1, arg762_1, buf883, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg1367_1
        del arg1368_1
        del arg761_1
        del arg762_1
        del buf882
        # Source Nodes: [x_986, x_987, x_988], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf884 = extern_kernels.convolution(buf883, arg763_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf884, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg763_1
        buf885 = buf883; del buf883  # reuse
        # Source Nodes: [x_990], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf884, buf885, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del buf884
        # Source Nodes: [x_990], Original ATen: [aten.convolution]
        buf886 = extern_kernels.convolution(buf885, arg764_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf886, (8, 864, 11, 11), (104544, 121, 11, 1))
        del arg764_1
        del buf885
        buf896 = reinterpret_tensor(buf899, (8, 864, 11, 11), (522720, 121, 11, 1), 104544)  # alias
        # Source Nodes: [x_comb_iter_1_left_13, x_comb_iter_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf861, arg1352_1, arg1353_1, arg741_1, arg742_1, buf844, buf896, 836352, grid=grid(836352), stream=stream0)
        del arg1352_1
        del arg1353_1
        del arg741_1
        del arg742_1
        del buf844
        del buf861
        buf897 = reinterpret_tensor(buf899, (8, 864, 11, 11), (522720, 121, 11, 1), 313632)  # alias
        # Source Nodes: [x_comb_iter_3_left_13, x_comb_iter_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_83.run(buf886, arg1370_1, arg1371_1, arg765_1, arg766_1, buf845, buf897, 836352, grid=grid(836352), stream=stream0)
        del arg1370_1
        del arg1371_1
        del arg765_1
        del arg766_1
        del buf845
        del buf886
        buf900 = empty_strided((8, 4320, 1, 1), (4320, 1, 34560, 34560), device='cuda', dtype=torch.float32)
        buf901 = reinterpret_tensor(buf900, (8, 4320, 1, 1), (4320, 1, 1, 1), 0); del buf900  # reuse
        # Source Nodes: [x_1003, x_1004], Original ATen: [aten.mean, aten.relu]
        triton_per_fused_mean_relu_85.run(buf901, buf899, 34560, 121, grid=grid(34560), stream=stream0)
        del buf878
        del buf895
        del buf896
        del buf897
        del buf898
        del buf899
        buf902 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1008], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg776_1, reinterpret_tensor(buf901, (8, 4320), (4320, 1), 0), reinterpret_tensor(arg775_1, (4320, 1000), (1, 4320), 0), alpha=1, beta=1, out=buf902)
        del arg775_1
        del arg776_1
        return (buf902, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((54, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((108, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((54, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((108, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((216, 540, 1, 1), (540, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((216, 540, 1, 1), (540, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((432, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((432, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((864, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((864, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((1000, 4320), (4320, 1), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg782_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg785_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg788_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg791_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg794_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg797_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg800_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg803_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg806_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg809_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg812_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg815_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg818_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg821_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg824_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg827_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg830_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg833_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg836_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg839_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg842_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg845_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg848_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg851_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg854_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg857_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg860_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg863_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg866_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg869_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg872_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg875_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg878_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg881_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg884_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg887_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg890_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg893_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg896_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg899_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg902_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg905_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg908_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg911_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg914_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg917_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg920_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg923_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg926_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg929_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg932_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg935_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg938_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg941_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg944_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg947_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg950_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg953_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg956_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg959_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg962_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg965_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg968_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg971_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg974_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg977_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg980_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg983_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg986_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg989_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg992_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg995_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg998_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1001_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1004_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1007_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1010_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1013_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1016_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1019_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1022_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1025_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1028_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1031_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1034_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1037_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1040_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1043_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1046_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1049_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1052_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1055_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1058_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1061_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1064_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1067_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1070_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1073_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1076_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1079_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1082_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1085_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1088_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1091_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1094_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1097_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1100_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1103_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1106_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1109_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1112_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1113_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1115_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1118_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1119_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1120_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1121_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1122_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1123_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1124_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1125_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1126_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1127_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1128_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1129_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1130_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1131_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1132_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1133_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1134_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1135_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1136_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1137_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1138_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1139_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1140_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1141_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1142_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1143_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1144_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1145_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1146_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1147_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1148_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1149_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1150_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1151_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1152_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1153_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1154_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1155_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1156_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1157_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1158_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1159_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1160_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1161_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1162_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1163_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1164_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1165_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1166_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1167_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1168_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1169_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1170_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1171_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1172_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1173_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1174_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1175_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1176_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1177_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1178_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1179_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1180_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1181_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1182_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1183_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1184_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1185_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1186_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1187_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1188_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1189_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1190_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1191_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1192_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1193_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1194_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1195_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1196_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1197_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1198_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1199_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1200_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1201_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1202_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1203_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1204_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1205_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1206_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1207_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1208_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1209_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1210_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1211_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1212_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1213_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1214_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1215_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1216_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1217_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1218_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1219_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1220_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1221_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1222_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1223_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1224_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1225_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1226_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1227_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1228_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1229_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1230_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1231_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1232_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1233_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1234_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1235_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1236_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1237_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1238_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1239_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1240_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1241_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1242_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1243_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1244_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1245_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1246_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1247_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1248_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1249_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1250_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1251_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1252_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1253_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1254_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1255_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1256_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1257_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1258_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1259_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1260_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1261_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1262_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1263_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1264_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1265_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1266_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1267_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1268_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1269_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1270_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1271_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1272_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1273_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1274_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1275_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1276_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1277_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1278_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1279_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1280_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1281_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1282_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1283_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1284_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1285_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1286_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1287_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1288_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1289_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1290_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1291_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1292_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1293_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1294_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1295_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1296_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1297_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1298_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1299_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1300_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1301_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1302_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1303_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1304_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1305_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1306_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1307_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1308_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1309_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1310_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1311_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1312_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1313_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1314_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1315_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1316_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1317_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1318_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1319_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1320_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1321_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1322_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1323_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1324_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1325_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1326_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1327_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1328_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1329_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1330_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1331_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1332_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1333_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1334_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1335_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1336_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1337_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1338_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1339_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1340_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1341_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1342_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1343_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1344_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1345_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1346_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1347_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1348_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1349_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1350_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1351_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1352_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1353_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1354_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1355_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1356_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1357_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1358_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1359_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1360_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1361_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1362_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1363_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1364_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1365_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1366_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1367_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1368_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1369_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1370_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1371_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1372_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1373_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1374_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1375_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1376_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1377_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1378_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1379_1 = rand_strided((8, 3, 331, 331), (328683, 109561, 331, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pnasnet5large', benchmark_compiled_module)
