
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


# kernel path: /tmp/torchinductor_youkaichao/ob/cobkh23eeokh7kldfxwa3llomuyupat5rbqbzjuf5sdufrnt3xqt.py
# Source Nodes: [l__mod___conv1_1, l__mod___conv1_2, l__mod___conv1_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___conv1_1 => add_1, mul_1, mul_2, sub
# l__mod___conv1_2 => relu
# l__mod___conv1_3 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/hk/chksp7rqminmauhx7wzbvwoqzhida4i3k42mv6kmk5vtcbkk2uvr.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_5, mul_7, mul_8, sub_2
# x_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 128
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogh3qpkvdhio7vdwg2f332mdm35byfewh72qtujbxv3z7a5rkir.py
# Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# shortcut => max_pool2d_with_indices
# x_1 => add_5, mul_7, mul_8, sub_2
# x_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-129) + (2*x0) + (256*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-128) + (2*x0) + (256*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-127) + (2*x0) + (256*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (256*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (256*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (256*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (127 + (2*x0) + (256*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (128 + (2*x0) + (256*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (129 + (2*x0) + (256*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (x4), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfn332dgiblg6b5ege6lqygpb6yw56zvzt5uhppjnzw47c5jedz.py
# Source Nodes: [out_1, out_2, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_1 => add_7, mul_10, mul_11, sub_3
# out_2 => relu_3
# x_4 => convolution_4
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cim33yixvhnhawtvcfeol3mniwf4xjkgdthrtw56mok3z6cgoznj.py
# Source Nodes: [x_5, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_5 => add_9, mul_13, mul_14, sub_4
# x_7 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2u/c2uqsrbvq3npvy77wl4gbeh6bonmcy5ynl6axb45unprabbpp4hk.py
# Source Nodes: [x_gap, x_gap_1, x_gap_2], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap => sum_1
# x_gap_1 => mean
# x_gap_2 => convolution_5
triton_red_fused_convolution_mean_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_sum_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (262144 + r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3grzqz2lytvueq6gzlc24xersg3yxvyw2mapk7de3mptfqqvvb.py
# Source Nodes: [x_attn, x_gap, x_gap_1, x_gap_2, x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
# x_attn => convolution_6
# x_gap => sum_1
# x_gap_1 => mean
# x_gap_2 => convolution_5
# x_gap_3 => add_11, mul_16, mul_17, sub_5
# x_gap_4 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
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


# kernel path: /tmp/torchinductor_youkaichao/ru/cruahjatuvup34ft7hci32hi2qwvk5ohud4lc4zk2qv3h4fs5ye2.py
# Source Nodes: [x_10], Original ATen: [aten._softmax]
# x_10 => amax, exp, sub_6
triton_poi_fused__softmax_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 128
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2p3gdnpmwzj44h4ukcc7wugwrnxqfrfltk64del3xrs43jxp7d.py
# Source Nodes: [mul, out_3, out_8], Original ATen: [aten.convolution, aten.mul, aten.sum]
# mul => mul_18
# out_3 => sum_3
# out_8 => convolution_7
triton_poi_fused_convolution_mul_sum_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 4096) % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (64 + x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cioidhogcucwu6le3mlckzj63gqmmvc45lgru5t6emtmuigiljgu.py
# Source Nodes: [out_10, out_9, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_10 => add_16
# out_9 => add_13, mul_20, mul_21, sub_7
# shortcut_1 => add_15, mul_23, mul_24, sub_8
# shortcut_2 => relu_6
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coewu6x4nticqh56zn2b3nvof4kwzy23sb3slqjn4pp42zmskszw.py
# Source Nodes: [out_21, out_22, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_21 => add_24, mul_36, mul_37, sub_13
# out_22 => add_25
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cje24d26ur7hwvwsg7dl2ce5wwvjq47mzrnwcvqjseglf6xt5klm.py
# Source Nodes: [x_30, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_30 => add_38, mul_55, mul_56, sub_20
# x_32 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutyagmrgujbmxc6gmy7o5oh324lzqwcwobghlsza3rkuu3fdtwj.py
# Source Nodes: [x_gap_15, x_gap_16, x_gap_17], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_15 => sum_10
# x_gap_16 => mean_3
# x_gap_17 => convolution_21
triton_red_fused_convolution_mean_sum_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_sum_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (524288 + r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7i4w5kfpxkuerlllgyxrsydw3epzxitsklul4mncj5uldmpvmy.py
# Source Nodes: [x_attn_6, x_gap_15, x_gap_16, x_gap_17, x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
# x_attn_6 => convolution_22
# x_gap_15 => sum_10
# x_gap_16 => mean_3
# x_gap_17 => convolution_21
# x_gap_18 => add_40, mul_58, mul_59, sub_21
# x_gap_19 => relu_17
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/uu/cuua6bhuwpfjyy3sgze2t4ojkns6db2vhmrwuoxgve4q2bkpepsx.py
# Source Nodes: [x_35], Original ATen: [aten._softmax]
# x_35 => amax_3, exp_3, sub_22
triton_poi_fused__softmax_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 256
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (128 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2srydnwk2ppkcrccw4t6g44miyidyv56hbep7ftc2l5f5ubtrx.py
# Source Nodes: [mul_3, out_39], Original ATen: [aten.mul, aten.sum]
# mul_3 => mul_60
# out_39 => sum_12
triton_poi_fused_mul_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 524288)
    x3 = xindex % 524288
    x1 = (xindex // 4096) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (1048576*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (524288 + x3 + (1048576*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkiok3ahjxds5oggxmazcllmuvfgbur6zesxunk4nnwojlbh6ip.py
# Source Nodes: [mul_3, out_39, out_44], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
# mul_3 => mul_60
# out_39 => sum_12
# out_44 => avg_pool2d
triton_poi_fused_avg_pool2d_mul_sum_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 32
    x0 = xindex % 32
    x3 = (xindex // 32)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-65) + (2*x0) + (128*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-64) + (2*x0) + (128*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-63) + (2*x0) + (128*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (128*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (128*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (63 + (2*x0) + (128*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 65, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23z5xckmc7hadjasiyrnwhyuhsth3doqpzvragopohbax563kig.py
# Source Nodes: [getattr_l__mod___layer2___0___downsample_0, getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_l__mod___layer2___0___downsample_0 => avg_pool2d_1
# getattr_l__mod___layer2___0___downsample_1 => convolution_24
triton_poi_fused_avg_pool2d_convolution_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcp32t4pmowsox6soeower7zhpwhoo3gypy7sxahwa3bd7ivrqw.py
# Source Nodes: [out_46, out_47, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_46 => add_42, mul_62, mul_63, sub_23
# out_47 => add_45
# shortcut_5 => add_44, mul_65, mul_66, sub_24
# shortcut_6 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpmagpxla4ciyvlddfeswn3qvbu3ttkcd2yqsb5pg2guv7o2gpw.py
# Source Nodes: [out_50, out_51, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_50 => add_47, mul_68, mul_69, sub_25
# out_51 => relu_19
# x_37 => convolution_26
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlvhxn3roybbnwq3rpfwmmgap7rjjdohno4w36yswyjj2qnlwlb.py
# Source Nodes: [x_38, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_38 => add_49, mul_71, mul_72, sub_26
# x_40 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllznrpel2ubvb5pgjaufwdbahqrgkk54ab72sqcleboiarvwihq.py
# Source Nodes: [x_gap_20, x_gap_21, x_gap_22], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_20 => sum_13
# x_gap_21 => mean_4
# x_gap_22 => convolution_27
triton_per_fused_convolution_mean_sum_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (262144*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (1024*x0) + (262144*x1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftmki6do3tslsw2gae3riqzoh3v5kv4umfvfhz5xq4vpg4nr5yg.py
# Source Nodes: [mul_4, out_52, out_57], Original ATen: [aten.convolution, aten.mul, aten.sum]
# mul_4 => mul_76
# out_52 => sum_15
# out_57 => convolution_29
triton_poi_fused_convolution_mul_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 1024) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshfjvxluqnzpjwzlierocs4quo7vdj3dedsvklvwtcqk3godfen.py
# Source Nodes: [out_58, out_59, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_58 => add_53, mul_78, mul_79, sub_29
# out_59 => add_54
# shortcut_7 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphyqodqp346zjgokeme3sezp55enieirzrsela2gpdzu6ec4ga5.py
# Source Nodes: [out_82, out_83, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_82 => add_71, mul_104, mul_105, sub_39
# out_83 => add_72
# shortcut_9 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjq4o7m3k4repj6kvssjviuzf43z3kpgnh7bgyqqhltcmxzkxeq.py
# Source Nodes: [x_63, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_63 => add_76, mul_110, mul_111, sub_41
# x_65 => relu_32
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czddmomxbgf4eleda3tphaucrsvs4xpdpln5gmp6ch4lbn3i4jo5.py
# Source Nodes: [x_gap_35, x_gap_36, x_gap_37], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_35 => sum_22
# x_gap_36 => mean_7
# x_gap_37 => convolution_42
triton_per_fused_convolution_mean_sum_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (524288*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (262144 + r2 + (1024*x0) + (524288*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6uno2vuuie6b42cl5krzqbxv2zzmguq3y2rjbcfpc7e7uvkpqrb.py
# Source Nodes: [x_attn_14, x_gap_35, x_gap_36, x_gap_37, x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
# x_attn_14 => convolution_43
# x_gap_35 => sum_22
# x_gap_36 => mean_7
# x_gap_37 => convolution_42
# x_gap_38 => add_78, mul_113, mul_114, sub_42
# x_gap_39 => relu_33
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
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


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvjxjgzwzwhugmyn32o6aunfbbylpytbir2jgx5osrpiuglokpb.py
# Source Nodes: [x_68], Original ATen: [aten._softmax]
# x_68 => amax_7, exp_7, sub_43
triton_poi_fused__softmax_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 512
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (256 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46mdmvte4yjptwpukvjfcxoelg7psio2qcbl5rugtjc7ovfmeuq.py
# Source Nodes: [mul_7, out_88], Original ATen: [aten.mul, aten.sum]
# mul_7 => mul_115
# out_88 => sum_24
triton_poi_fused_mul_sum_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 1024) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csx6v4h67auysdz72oe47vb7aru2nbiynv4g63vlr3qpkdkau4n4.py
# Source Nodes: [mul_7, out_88, out_93], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
# mul_7 => mul_115
# out_88 => sum_24
# out_93 => avg_pool2d_2
triton_poi_fused_avg_pool2d_mul_sum_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x3 = (xindex // 16)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + (2*x0) + (64*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-32) + (2*x0) + (64*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-31) + (2*x0) + (64*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (64*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (64*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (31 + (2*x0) + (64*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 33, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4hsgtnsvfjxhl2jy53xzhx4z4ctb63kt4nk6qmoqrdgmjt22rg.py
# Source Nodes: [getattr_l__mod___layer3___0___downsample_0, getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_l__mod___layer3___0___downsample_0 => avg_pool2d_3
# getattr_l__mod___layer3___0___downsample_1 => convolution_45
triton_poi_fused_avg_pool2d_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgudi2ovune5cqyvvlzft64bkmuqjb7tan46ugxyqzg56hnrik4.py
# Source Nodes: [out_95, out_96, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_95 => add_80, mul_117, mul_118, sub_44
# out_96 => add_83
# shortcut_10 => add_82, mul_120, mul_121, sub_45
# shortcut_11 => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhh2jtil5u3ebezisfaijrivjiaabykq63vo732mjc5db2t4h2b.py
# Source Nodes: [out_100, out_99, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_100 => relu_35
# out_99 => add_85, mul_123, mul_124, sub_46
# x_70 => convolution_47
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cev4gv54e372z2fnxaxyan225povdw3l76vbbdzxy5kobtpu5n67.py
# Source Nodes: [x_71, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_71 => add_87, mul_126, mul_127, sub_47
# x_73 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c2372jpsdqlg4kqyjpwyrohymyc5bnnwux7czlg3h6mnc7ujqlhh.py
# Source Nodes: [x_gap_40, x_gap_41, x_gap_42], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_40 => sum_25
# x_gap_41 => mean_8
# x_gap_42 => convolution_48
triton_per_fused_convolution_mean_sum_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (131072*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (65536 + r2 + (256*x0) + (131072*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clz5hi4xl7p37ijihiw3ovz2cz4cttjaivo4ejvoirpwu6reu42x.py
# Source Nodes: [mul_8, out_101, out_106], Original ATen: [aten.convolution, aten.mul, aten.sum]
# mul_8 => mul_131
# out_101 => sum_27
# out_106 => convolution_50
triton_poi_fused_convolution_mul_sum_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 65536)
    x3 = xindex % 65536
    x1 = (xindex // 256) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (131072*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (65536 + x3 + (131072*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyom46vevi4p5bftyspsmmiuuzitzxfgd5kj45b3g4hqegw5b6w.py
# Source Nodes: [out_107, out_108, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_107 => add_91, mul_133, mul_134, sub_50
# out_108 => add_92
# shortcut_12 => relu_38
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedoxoydn37252aee4vhg6e3uso52f5zcwganoib4khkt5lvq7ol.py
# Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_248 => add_285, mul_412, mul_413, sub_157
# x_250 => relu_124
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ubw7dvum5ranfhbcc24pavbnesqdbb2xj5enddoiykthcpg72j.py
# Source Nodes: [x_gap_150, x_gap_151, x_gap_152], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_150 => sum_91
# x_gap_151 => mean_30
# x_gap_152 => convolution_158
triton_per_fused_convolution_mean_sum_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (262144*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (256*x0) + (262144*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbxiia52ambh64m2jok6sm65xqkt2bwt7dnqpswh2wo5kprzqmh.py
# Source Nodes: [x_attn_60, x_gap_150, x_gap_151, x_gap_152, x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
# x_attn_60 => convolution_159
# x_gap_150 => sum_91
# x_gap_151 => mean_30
# x_gap_152 => convolution_158
# x_gap_153 => add_287, mul_415, mul_416, sub_158
# x_gap_154 => relu_125
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5m/c5mkjjelextic4ithatekjkio5qkvblzglayuvsxyemlh5h2zmmo.py
# Source Nodes: [x_253], Original ATen: [aten._softmax]
# x_253 => amax_30, exp_30, sub_159
triton_poi_fused__softmax_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 1024
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qsl3mng3u2kyeegvdkdpzxhmhnnbdbp76uoavmatzlxgwnfyvc.py
# Source Nodes: [mul_30, out_365], Original ATen: [aten.mul, aten.sum]
# mul_30 => mul_417
# out_365 => sum_93
triton_poi_fused_mul_sum_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 256) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3rv4pbzjhbuop3dgoye3iz6rarxdyllgmi3guqgojls7s3yvqs.py
# Source Nodes: [mul_30, out_365, out_370], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
# mul_30 => mul_417
# out_365 => sum_93
# out_370 => avg_pool2d_4
triton_poi_fused_avg_pool2d_mul_sum_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8) % 8
    x0 = xindex % 8
    x3 = (xindex // 8)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + (2*x0) + (32*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16) + (2*x0) + (32*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-15) + (2*x0) + (32*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (32*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (32*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (15 + (2*x0) + (32*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 17, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lcnagkstsfwyhl7itcaf2auwl65i46qtgendqswp672irbdnhc.py
# Source Nodes: [getattr_l__mod___layer4___0___downsample_0, getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_l__mod___layer4___0___downsample_0 => avg_pool2d_5
# getattr_l__mod___layer4___0___downsample_1 => convolution_161
triton_poi_fused_avg_pool2d_convolution_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfnd7wqfjkuzahlwhualilukvmw26lmdp3rmssmmepdme6ysegr.py
# Source Nodes: [out_372, out_373, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_372 => add_289, mul_419, mul_420, sub_160
# out_373 => add_292
# shortcut_34 => add_291, mul_422, mul_423, sub_161
# shortcut_35 => relu_126
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqb4mvu6tq6coeo5fxuikdklsax6b2ah4432cfoqjitenpmexg5.py
# Source Nodes: [out_376, out_377, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_376 => add_294, mul_425, mul_426, sub_162
# out_377 => relu_127
# x_255 => convolution_163
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyqrkqyjyrxqkgu2dhvmeac5mfpnmb57mhebnab43snf2twsqz5.py
# Source Nodes: [x_256, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_256 => add_296, mul_428, mul_429, sub_163
# x_258 => relu_128
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1024
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4ui4bnhg4lngqwf3gooyp6k4ia57updcjkuolko4wpsezxciitp.py
# Source Nodes: [x_gap_155, x_gap_156, x_gap_157], Original ATen: [aten.convolution, aten.mean, aten.sum]
# x_gap_155 => sum_94
# x_gap_156 => mean_31
# x_gap_157 => convolution_164
triton_per_fused_convolution_mean_sum_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x0) + (65536*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32768 + r2 + (64*x0) + (65536*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunsigy7ebxwya53surn4wgdrqljo6lk353uidbhmtit5jykucac.py
# Source Nodes: [mul_31, out_378, out_383], Original ATen: [aten.convolution, aten.mul, aten.sum]
# mul_31 => mul_433
# out_378 => sum_96
# out_383 => convolution_166
triton_poi_fused_convolution_mul_sum_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 32768)
    x3 = xindex % 32768
    x1 = (xindex // 64) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32768 + x3 + (65536*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxcqkepj4cg7h2tjnql3zaismscg2lmgrvy52ok5t7bd4vziv44.py
# Source Nodes: [out_384, out_385, shortcut_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_384 => add_300, mul_435, mul_436, sub_166
# out_385 => add_301
# shortcut_36 => relu_130
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy65a5dxer6twsmylpe6277me4vglyd6m3qeko6brjpjjk6wl2fs.py
# Source Nodes: [out_396, out_397, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# out_396 => add_309, mul_448, mul_449, sub_171
# out_397 => add_310
# x_272 => relu_134
# x_273 => mean_33
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (r2 + (64*x3)), rmask, other=0.0)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 64.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (32, ), (1, ))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (32, ), (1, ))
    assert_size_stride(arg37_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (32, ), (1, ))
    assert_size_stride(arg51_1, (32, ), (1, ))
    assert_size_stride(arg52_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg82_1, (64, ), (1, ))
    assert_size_stride(arg83_1, (64, ), (1, ))
    assert_size_stride(arg84_1, (64, ), (1, ))
    assert_size_stride(arg85_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg97_1, (64, ), (1, ))
    assert_size_stride(arg98_1, (64, ), (1, ))
    assert_size_stride(arg99_1, (64, ), (1, ))
    assert_size_stride(arg100_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (128, ), (1, ))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg160_1, (128, ), (1, ))
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, ), (1, ))
    assert_size_stride(arg186_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (128, ), (1, ))
    assert_size_stride(arg192_1, (128, ), (1, ))
    assert_size_stride(arg193_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (256, ), (1, ))
    assert_size_stride(arg216_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (128, ), (1, ))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (128, ), (1, ))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg274_1, (256, ), (1, ))
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg277_1, (512, ), (1, ))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (128, ), (1, ))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg289_1, (256, ), (1, ))
    assert_size_stride(arg290_1, (256, ), (1, ))
    assert_size_stride(arg291_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg295_1, (128, ), (1, ))
    assert_size_stride(arg296_1, (128, ), (1, ))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg304_1, (256, ), (1, ))
    assert_size_stride(arg305_1, (256, ), (1, ))
    assert_size_stride(arg306_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, ), (1, ))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg319_1, (256, ), (1, ))
    assert_size_stride(arg320_1, (256, ), (1, ))
    assert_size_stride(arg321_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg325_1, (128, ), (1, ))
    assert_size_stride(arg326_1, (128, ), (1, ))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg334_1, (256, ), (1, ))
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (128, ), (1, ))
    assert_size_stride(arg342_1, (128, ), (1, ))
    assert_size_stride(arg343_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg344_1, (512, ), (1, ))
    assert_size_stride(arg345_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg349_1, (256, ), (1, ))
    assert_size_stride(arg350_1, (256, ), (1, ))
    assert_size_stride(arg351_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg352_1, (512, ), (1, ))
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg359_1, (512, ), (1, ))
    assert_size_stride(arg360_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg364_1, (256, ), (1, ))
    assert_size_stride(arg365_1, (256, ), (1, ))
    assert_size_stride(arg366_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg367_1, (512, ), (1, ))
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (128, ), (1, ))
    assert_size_stride(arg373_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg379_1, (256, ), (1, ))
    assert_size_stride(arg380_1, (256, ), (1, ))
    assert_size_stride(arg381_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg382_1, (512, ), (1, ))
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg385_1, (128, ), (1, ))
    assert_size_stride(arg386_1, (128, ), (1, ))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg389_1, (512, ), (1, ))
    assert_size_stride(arg390_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg394_1, (256, ), (1, ))
    assert_size_stride(arg395_1, (256, ), (1, ))
    assert_size_stride(arg396_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg397_1, (512, ), (1, ))
    assert_size_stride(arg398_1, (512, ), (1, ))
    assert_size_stride(arg399_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg400_1, (128, ), (1, ))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (128, ), (1, ))
    assert_size_stride(arg403_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg404_1, (512, ), (1, ))
    assert_size_stride(arg405_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg406_1, (1024, ), (1, ))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg409_1, (256, ), (1, ))
    assert_size_stride(arg410_1, (256, ), (1, ))
    assert_size_stride(arg411_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg412_1, (512, ), (1, ))
    assert_size_stride(arg413_1, (512, ), (1, ))
    assert_size_stride(arg414_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg415_1, (128, ), (1, ))
    assert_size_stride(arg416_1, (128, ), (1, ))
    assert_size_stride(arg417_1, (128, ), (1, ))
    assert_size_stride(arg418_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg419_1, (512, ), (1, ))
    assert_size_stride(arg420_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg424_1, (256, ), (1, ))
    assert_size_stride(arg425_1, (256, ), (1, ))
    assert_size_stride(arg426_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg427_1, (512, ), (1, ))
    assert_size_stride(arg428_1, (512, ), (1, ))
    assert_size_stride(arg429_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg430_1, (128, ), (1, ))
    assert_size_stride(arg431_1, (128, ), (1, ))
    assert_size_stride(arg432_1, (128, ), (1, ))
    assert_size_stride(arg433_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg434_1, (512, ), (1, ))
    assert_size_stride(arg435_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg439_1, (256, ), (1, ))
    assert_size_stride(arg440_1, (256, ), (1, ))
    assert_size_stride(arg441_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg442_1, (512, ), (1, ))
    assert_size_stride(arg443_1, (512, ), (1, ))
    assert_size_stride(arg444_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg445_1, (128, ), (1, ))
    assert_size_stride(arg446_1, (128, ), (1, ))
    assert_size_stride(arg447_1, (128, ), (1, ))
    assert_size_stride(arg448_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg454_1, (256, ), (1, ))
    assert_size_stride(arg455_1, (256, ), (1, ))
    assert_size_stride(arg456_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg457_1, (512, ), (1, ))
    assert_size_stride(arg458_1, (512, ), (1, ))
    assert_size_stride(arg459_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (128, ), (1, ))
    assert_size_stride(arg462_1, (128, ), (1, ))
    assert_size_stride(arg463_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg464_1, (512, ), (1, ))
    assert_size_stride(arg465_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg469_1, (512, ), (1, ))
    assert_size_stride(arg470_1, (512, ), (1, ))
    assert_size_stride(arg471_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg475_1, (256, ), (1, ))
    assert_size_stride(arg476_1, (256, ), (1, ))
    assert_size_stride(arg477_1, (256, ), (1, ))
    assert_size_stride(arg478_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg481_1, (2048, ), (1, ))
    assert_size_stride(arg482_1, (2048, ), (1, ))
    assert_size_stride(arg483_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg484_1, (2048, ), (1, ))
    assert_size_stride(arg485_1, (2048, ), (1, ))
    assert_size_stride(arg486_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg487_1, (512, ), (1, ))
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg493_1, (256, ), (1, ))
    assert_size_stride(arg494_1, (256, ), (1, ))
    assert_size_stride(arg495_1, (256, ), (1, ))
    assert_size_stride(arg496_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (2048, ), (1, ))
    assert_size_stride(arg501_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (1024, ), (1, ))
    assert_size_stride(arg507_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg508_1, (256, ), (1, ))
    assert_size_stride(arg509_1, (256, ), (1, ))
    assert_size_stride(arg510_1, (256, ), (1, ))
    assert_size_stride(arg511_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg514_1, (2048, ), (1, ))
    assert_size_stride(arg515_1, (2048, ), (1, ))
    assert_size_stride(arg516_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg517_1, (1000, ), (1, ))
    assert_size_stride(arg518_1, (64, ), (1, ))
    assert_size_stride(arg519_1, (64, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (64, ), (1, ))
    assert_size_stride(arg522_1, (64, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (128, ), (1, ))
    assert_size_stride(arg525_1, (128, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (64, ), (1, ))
    assert_size_stride(arg528_1, (64, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (128, ), (1, ))
    assert_size_stride(arg531_1, (128, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (32, ), (1, ))
    assert_size_stride(arg534_1, (32, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (256, ), (1, ))
    assert_size_stride(arg537_1, (256, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (256, ), (1, ))
    assert_size_stride(arg540_1, (256, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (64, ), (1, ))
    assert_size_stride(arg543_1, (64, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (128, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (32, ), (1, ))
    assert_size_stride(arg549_1, (32, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (256, ), (1, ))
    assert_size_stride(arg552_1, (256, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (64, ), (1, ))
    assert_size_stride(arg555_1, (64, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (128, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (32, ), (1, ))
    assert_size_stride(arg561_1, (32, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (256, ), (1, ))
    assert_size_stride(arg564_1, (256, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (128, ), (1, ))
    assert_size_stride(arg567_1, (128, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (256, ), (1, ))
    assert_size_stride(arg570_1, (256, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (64, ), (1, ))
    assert_size_stride(arg573_1, (64, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (512, ), (1, ))
    assert_size_stride(arg576_1, (512, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (512, ), (1, ))
    assert_size_stride(arg579_1, (512, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (128, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (256, ), (1, ))
    assert_size_stride(arg585_1, (256, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (64, ), (1, ))
    assert_size_stride(arg588_1, (64, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (512, ), (1, ))
    assert_size_stride(arg591_1, (512, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (128, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (256, ), (1, ))
    assert_size_stride(arg597_1, (256, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (64, ), (1, ))
    assert_size_stride(arg600_1, (64, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (512, ), (1, ))
    assert_size_stride(arg603_1, (512, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (128, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (256, ), (1, ))
    assert_size_stride(arg609_1, (256, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (64, ), (1, ))
    assert_size_stride(arg612_1, (64, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (512, ), (1, ))
    assert_size_stride(arg615_1, (512, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (256, ), (1, ))
    assert_size_stride(arg618_1, (256, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (512, ), (1, ))
    assert_size_stride(arg621_1, (512, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (128, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (1024, ), (1, ))
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (1024, ), (1, ))
    assert_size_stride(arg630_1, (1024, ), (1, ))
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (256, ), (1, ))
    assert_size_stride(arg633_1, (256, ), (1, ))
    assert_size_stride(arg634_1, (), ())
    assert_size_stride(arg635_1, (512, ), (1, ))
    assert_size_stride(arg636_1, (512, ), (1, ))
    assert_size_stride(arg637_1, (), ())
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (128, ), (1, ))
    assert_size_stride(arg640_1, (), ())
    assert_size_stride(arg641_1, (1024, ), (1, ))
    assert_size_stride(arg642_1, (1024, ), (1, ))
    assert_size_stride(arg643_1, (), ())
    assert_size_stride(arg644_1, (256, ), (1, ))
    assert_size_stride(arg645_1, (256, ), (1, ))
    assert_size_stride(arg646_1, (), ())
    assert_size_stride(arg647_1, (512, ), (1, ))
    assert_size_stride(arg648_1, (512, ), (1, ))
    assert_size_stride(arg649_1, (), ())
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (128, ), (1, ))
    assert_size_stride(arg652_1, (), ())
    assert_size_stride(arg653_1, (1024, ), (1, ))
    assert_size_stride(arg654_1, (1024, ), (1, ))
    assert_size_stride(arg655_1, (), ())
    assert_size_stride(arg656_1, (256, ), (1, ))
    assert_size_stride(arg657_1, (256, ), (1, ))
    assert_size_stride(arg658_1, (), ())
    assert_size_stride(arg659_1, (512, ), (1, ))
    assert_size_stride(arg660_1, (512, ), (1, ))
    assert_size_stride(arg661_1, (), ())
    assert_size_stride(arg662_1, (128, ), (1, ))
    assert_size_stride(arg663_1, (128, ), (1, ))
    assert_size_stride(arg664_1, (), ())
    assert_size_stride(arg665_1, (1024, ), (1, ))
    assert_size_stride(arg666_1, (1024, ), (1, ))
    assert_size_stride(arg667_1, (), ())
    assert_size_stride(arg668_1, (256, ), (1, ))
    assert_size_stride(arg669_1, (256, ), (1, ))
    assert_size_stride(arg670_1, (), ())
    assert_size_stride(arg671_1, (512, ), (1, ))
    assert_size_stride(arg672_1, (512, ), (1, ))
    assert_size_stride(arg673_1, (), ())
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (128, ), (1, ))
    assert_size_stride(arg676_1, (), ())
    assert_size_stride(arg677_1, (1024, ), (1, ))
    assert_size_stride(arg678_1, (1024, ), (1, ))
    assert_size_stride(arg679_1, (), ())
    assert_size_stride(arg680_1, (256, ), (1, ))
    assert_size_stride(arg681_1, (256, ), (1, ))
    assert_size_stride(arg682_1, (), ())
    assert_size_stride(arg683_1, (512, ), (1, ))
    assert_size_stride(arg684_1, (512, ), (1, ))
    assert_size_stride(arg685_1, (), ())
    assert_size_stride(arg686_1, (128, ), (1, ))
    assert_size_stride(arg687_1, (128, ), (1, ))
    assert_size_stride(arg688_1, (), ())
    assert_size_stride(arg689_1, (1024, ), (1, ))
    assert_size_stride(arg690_1, (1024, ), (1, ))
    assert_size_stride(arg691_1, (), ())
    assert_size_stride(arg692_1, (256, ), (1, ))
    assert_size_stride(arg693_1, (256, ), (1, ))
    assert_size_stride(arg694_1, (), ())
    assert_size_stride(arg695_1, (512, ), (1, ))
    assert_size_stride(arg696_1, (512, ), (1, ))
    assert_size_stride(arg697_1, (), ())
    assert_size_stride(arg698_1, (128, ), (1, ))
    assert_size_stride(arg699_1, (128, ), (1, ))
    assert_size_stride(arg700_1, (), ())
    assert_size_stride(arg701_1, (1024, ), (1, ))
    assert_size_stride(arg702_1, (1024, ), (1, ))
    assert_size_stride(arg703_1, (), ())
    assert_size_stride(arg704_1, (256, ), (1, ))
    assert_size_stride(arg705_1, (256, ), (1, ))
    assert_size_stride(arg706_1, (), ())
    assert_size_stride(arg707_1, (512, ), (1, ))
    assert_size_stride(arg708_1, (512, ), (1, ))
    assert_size_stride(arg709_1, (), ())
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (128, ), (1, ))
    assert_size_stride(arg712_1, (), ())
    assert_size_stride(arg713_1, (1024, ), (1, ))
    assert_size_stride(arg714_1, (1024, ), (1, ))
    assert_size_stride(arg715_1, (), ())
    assert_size_stride(arg716_1, (256, ), (1, ))
    assert_size_stride(arg717_1, (256, ), (1, ))
    assert_size_stride(arg718_1, (), ())
    assert_size_stride(arg719_1, (512, ), (1, ))
    assert_size_stride(arg720_1, (512, ), (1, ))
    assert_size_stride(arg721_1, (), ())
    assert_size_stride(arg722_1, (128, ), (1, ))
    assert_size_stride(arg723_1, (128, ), (1, ))
    assert_size_stride(arg724_1, (), ())
    assert_size_stride(arg725_1, (1024, ), (1, ))
    assert_size_stride(arg726_1, (1024, ), (1, ))
    assert_size_stride(arg727_1, (), ())
    assert_size_stride(arg728_1, (256, ), (1, ))
    assert_size_stride(arg729_1, (256, ), (1, ))
    assert_size_stride(arg730_1, (), ())
    assert_size_stride(arg731_1, (512, ), (1, ))
    assert_size_stride(arg732_1, (512, ), (1, ))
    assert_size_stride(arg733_1, (), ())
    assert_size_stride(arg734_1, (128, ), (1, ))
    assert_size_stride(arg735_1, (128, ), (1, ))
    assert_size_stride(arg736_1, (), ())
    assert_size_stride(arg737_1, (1024, ), (1, ))
    assert_size_stride(arg738_1, (1024, ), (1, ))
    assert_size_stride(arg739_1, (), ())
    assert_size_stride(arg740_1, (256, ), (1, ))
    assert_size_stride(arg741_1, (256, ), (1, ))
    assert_size_stride(arg742_1, (), ())
    assert_size_stride(arg743_1, (512, ), (1, ))
    assert_size_stride(arg744_1, (512, ), (1, ))
    assert_size_stride(arg745_1, (), ())
    assert_size_stride(arg746_1, (128, ), (1, ))
    assert_size_stride(arg747_1, (128, ), (1, ))
    assert_size_stride(arg748_1, (), ())
    assert_size_stride(arg749_1, (1024, ), (1, ))
    assert_size_stride(arg750_1, (1024, ), (1, ))
    assert_size_stride(arg751_1, (), ())
    assert_size_stride(arg752_1, (256, ), (1, ))
    assert_size_stride(arg753_1, (256, ), (1, ))
    assert_size_stride(arg754_1, (), ())
    assert_size_stride(arg755_1, (512, ), (1, ))
    assert_size_stride(arg756_1, (512, ), (1, ))
    assert_size_stride(arg757_1, (), ())
    assert_size_stride(arg758_1, (128, ), (1, ))
    assert_size_stride(arg759_1, (128, ), (1, ))
    assert_size_stride(arg760_1, (), ())
    assert_size_stride(arg761_1, (1024, ), (1, ))
    assert_size_stride(arg762_1, (1024, ), (1, ))
    assert_size_stride(arg763_1, (), ())
    assert_size_stride(arg764_1, (256, ), (1, ))
    assert_size_stride(arg765_1, (256, ), (1, ))
    assert_size_stride(arg766_1, (), ())
    assert_size_stride(arg767_1, (512, ), (1, ))
    assert_size_stride(arg768_1, (512, ), (1, ))
    assert_size_stride(arg769_1, (), ())
    assert_size_stride(arg770_1, (128, ), (1, ))
    assert_size_stride(arg771_1, (128, ), (1, ))
    assert_size_stride(arg772_1, (), ())
    assert_size_stride(arg773_1, (1024, ), (1, ))
    assert_size_stride(arg774_1, (1024, ), (1, ))
    assert_size_stride(arg775_1, (), ())
    assert_size_stride(arg776_1, (256, ), (1, ))
    assert_size_stride(arg777_1, (256, ), (1, ))
    assert_size_stride(arg778_1, (), ())
    assert_size_stride(arg779_1, (512, ), (1, ))
    assert_size_stride(arg780_1, (512, ), (1, ))
    assert_size_stride(arg781_1, (), ())
    assert_size_stride(arg782_1, (128, ), (1, ))
    assert_size_stride(arg783_1, (128, ), (1, ))
    assert_size_stride(arg784_1, (), ())
    assert_size_stride(arg785_1, (1024, ), (1, ))
    assert_size_stride(arg786_1, (1024, ), (1, ))
    assert_size_stride(arg787_1, (), ())
    assert_size_stride(arg788_1, (256, ), (1, ))
    assert_size_stride(arg789_1, (256, ), (1, ))
    assert_size_stride(arg790_1, (), ())
    assert_size_stride(arg791_1, (512, ), (1, ))
    assert_size_stride(arg792_1, (512, ), (1, ))
    assert_size_stride(arg793_1, (), ())
    assert_size_stride(arg794_1, (128, ), (1, ))
    assert_size_stride(arg795_1, (128, ), (1, ))
    assert_size_stride(arg796_1, (), ())
    assert_size_stride(arg797_1, (1024, ), (1, ))
    assert_size_stride(arg798_1, (1024, ), (1, ))
    assert_size_stride(arg799_1, (), ())
    assert_size_stride(arg800_1, (256, ), (1, ))
    assert_size_stride(arg801_1, (256, ), (1, ))
    assert_size_stride(arg802_1, (), ())
    assert_size_stride(arg803_1, (512, ), (1, ))
    assert_size_stride(arg804_1, (512, ), (1, ))
    assert_size_stride(arg805_1, (), ())
    assert_size_stride(arg806_1, (128, ), (1, ))
    assert_size_stride(arg807_1, (128, ), (1, ))
    assert_size_stride(arg808_1, (), ())
    assert_size_stride(arg809_1, (1024, ), (1, ))
    assert_size_stride(arg810_1, (1024, ), (1, ))
    assert_size_stride(arg811_1, (), ())
    assert_size_stride(arg812_1, (256, ), (1, ))
    assert_size_stride(arg813_1, (256, ), (1, ))
    assert_size_stride(arg814_1, (), ())
    assert_size_stride(arg815_1, (512, ), (1, ))
    assert_size_stride(arg816_1, (512, ), (1, ))
    assert_size_stride(arg817_1, (), ())
    assert_size_stride(arg818_1, (128, ), (1, ))
    assert_size_stride(arg819_1, (128, ), (1, ))
    assert_size_stride(arg820_1, (), ())
    assert_size_stride(arg821_1, (1024, ), (1, ))
    assert_size_stride(arg822_1, (1024, ), (1, ))
    assert_size_stride(arg823_1, (), ())
    assert_size_stride(arg824_1, (256, ), (1, ))
    assert_size_stride(arg825_1, (256, ), (1, ))
    assert_size_stride(arg826_1, (), ())
    assert_size_stride(arg827_1, (512, ), (1, ))
    assert_size_stride(arg828_1, (512, ), (1, ))
    assert_size_stride(arg829_1, (), ())
    assert_size_stride(arg830_1, (128, ), (1, ))
    assert_size_stride(arg831_1, (128, ), (1, ))
    assert_size_stride(arg832_1, (), ())
    assert_size_stride(arg833_1, (1024, ), (1, ))
    assert_size_stride(arg834_1, (1024, ), (1, ))
    assert_size_stride(arg835_1, (), ())
    assert_size_stride(arg836_1, (256, ), (1, ))
    assert_size_stride(arg837_1, (256, ), (1, ))
    assert_size_stride(arg838_1, (), ())
    assert_size_stride(arg839_1, (512, ), (1, ))
    assert_size_stride(arg840_1, (512, ), (1, ))
    assert_size_stride(arg841_1, (), ())
    assert_size_stride(arg842_1, (128, ), (1, ))
    assert_size_stride(arg843_1, (128, ), (1, ))
    assert_size_stride(arg844_1, (), ())
    assert_size_stride(arg845_1, (1024, ), (1, ))
    assert_size_stride(arg846_1, (1024, ), (1, ))
    assert_size_stride(arg847_1, (), ())
    assert_size_stride(arg848_1, (256, ), (1, ))
    assert_size_stride(arg849_1, (256, ), (1, ))
    assert_size_stride(arg850_1, (), ())
    assert_size_stride(arg851_1, (512, ), (1, ))
    assert_size_stride(arg852_1, (512, ), (1, ))
    assert_size_stride(arg853_1, (), ())
    assert_size_stride(arg854_1, (128, ), (1, ))
    assert_size_stride(arg855_1, (128, ), (1, ))
    assert_size_stride(arg856_1, (), ())
    assert_size_stride(arg857_1, (1024, ), (1, ))
    assert_size_stride(arg858_1, (1024, ), (1, ))
    assert_size_stride(arg859_1, (), ())
    assert_size_stride(arg860_1, (256, ), (1, ))
    assert_size_stride(arg861_1, (256, ), (1, ))
    assert_size_stride(arg862_1, (), ())
    assert_size_stride(arg863_1, (512, ), (1, ))
    assert_size_stride(arg864_1, (512, ), (1, ))
    assert_size_stride(arg865_1, (), ())
    assert_size_stride(arg866_1, (128, ), (1, ))
    assert_size_stride(arg867_1, (128, ), (1, ))
    assert_size_stride(arg868_1, (), ())
    assert_size_stride(arg869_1, (1024, ), (1, ))
    assert_size_stride(arg870_1, (1024, ), (1, ))
    assert_size_stride(arg871_1, (), ())
    assert_size_stride(arg872_1, (256, ), (1, ))
    assert_size_stride(arg873_1, (256, ), (1, ))
    assert_size_stride(arg874_1, (), ())
    assert_size_stride(arg875_1, (512, ), (1, ))
    assert_size_stride(arg876_1, (512, ), (1, ))
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (128, ), (1, ))
    assert_size_stride(arg879_1, (128, ), (1, ))
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (1024, ), (1, ))
    assert_size_stride(arg882_1, (1024, ), (1, ))
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (256, ), (1, ))
    assert_size_stride(arg885_1, (256, ), (1, ))
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (512, ), (1, ))
    assert_size_stride(arg888_1, (512, ), (1, ))
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (128, ), (1, ))
    assert_size_stride(arg891_1, (128, ), (1, ))
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (1024, ), (1, ))
    assert_size_stride(arg894_1, (1024, ), (1, ))
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (512, ), (1, ))
    assert_size_stride(arg897_1, (512, ), (1, ))
    assert_size_stride(arg898_1, (), ())
    assert_size_stride(arg899_1, (1024, ), (1, ))
    assert_size_stride(arg900_1, (1024, ), (1, ))
    assert_size_stride(arg901_1, (), ())
    assert_size_stride(arg902_1, (256, ), (1, ))
    assert_size_stride(arg903_1, (256, ), (1, ))
    assert_size_stride(arg904_1, (), ())
    assert_size_stride(arg905_1, (2048, ), (1, ))
    assert_size_stride(arg906_1, (2048, ), (1, ))
    assert_size_stride(arg907_1, (), ())
    assert_size_stride(arg908_1, (2048, ), (1, ))
    assert_size_stride(arg909_1, (2048, ), (1, ))
    assert_size_stride(arg910_1, (), ())
    assert_size_stride(arg911_1, (512, ), (1, ))
    assert_size_stride(arg912_1, (512, ), (1, ))
    assert_size_stride(arg913_1, (), ())
    assert_size_stride(arg914_1, (1024, ), (1, ))
    assert_size_stride(arg915_1, (1024, ), (1, ))
    assert_size_stride(arg916_1, (), ())
    assert_size_stride(arg917_1, (256, ), (1, ))
    assert_size_stride(arg918_1, (256, ), (1, ))
    assert_size_stride(arg919_1, (), ())
    assert_size_stride(arg920_1, (2048, ), (1, ))
    assert_size_stride(arg921_1, (2048, ), (1, ))
    assert_size_stride(arg922_1, (), ())
    assert_size_stride(arg923_1, (512, ), (1, ))
    assert_size_stride(arg924_1, (512, ), (1, ))
    assert_size_stride(arg925_1, (), ())
    assert_size_stride(arg926_1, (1024, ), (1, ))
    assert_size_stride(arg927_1, (1024, ), (1, ))
    assert_size_stride(arg928_1, (), ())
    assert_size_stride(arg929_1, (256, ), (1, ))
    assert_size_stride(arg930_1, (256, ), (1, ))
    assert_size_stride(arg931_1, (), ())
    assert_size_stride(arg932_1, (2048, ), (1, ))
    assert_size_stride(arg933_1, (2048, ), (1, ))
    assert_size_stride(arg934_1, (), ())
    assert_size_stride(arg935_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg935_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg0_1
        del arg935_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___conv1_1, l__mod___conv1_2, l__mod___conv1_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg518_1, arg519_1, arg1_1, arg2_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1_1
        del arg2_1
        del arg518_1
        del arg519_1
        # Source Nodes: [l__mod___conv1_1, l__mod___conv1_2, l__mod___conv1_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg3_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___conv1_4, l__mod___conv1_5, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf3, arg521_1, arg522_1, arg4_1, arg5_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg4_1
        del arg521_1
        del arg522_1
        del arg5_1
        # Source Nodes: [l__mod___conv1_4, l__mod___conv1_5, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        del arg6_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, arg524_1, arg525_1, arg7_1, arg8_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg524_1
        del arg525_1
        del arg7_1
        del arg8_1
        buf6 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2.run(buf5, buf6, 4194304, grid=grid(4194304), stream=stream0)
        del buf5
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg9_1
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [out_1, out_2, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf8, arg527_1, arg528_1, arg10_1, arg11_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg10_1
        del arg11_1
        del arg527_1
        del arg528_1
        # Source Nodes: [out_1, out_2, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf9 = extern_kernels.convolution(buf8, arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf9, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg12_1
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [x_5, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf10, arg530_1, arg531_1, arg13_1, arg14_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg13_1
        del arg14_1
        del arg530_1
        del arg531_1
        buf11 = empty_strided((8, 64, 1, 1), (64, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf12 = reinterpret_tensor(buf11, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf11  # reuse
        # Source Nodes: [x_gap, x_gap_1, x_gap_2], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_red_fused_convolution_mean_sum_5.run(buf12, buf10, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap, x_gap_1, x_gap_2], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf13 = extern_kernels.convolution(buf12, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg15_1
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [x_attn, x_gap, x_gap_1, x_gap_2, x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf14, arg16_1, arg533_1, arg534_1, arg17_1, arg18_1, 256, grid=grid(256), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del arg533_1
        del arg534_1
        # Source Nodes: [x_attn, x_gap, x_gap_1, x_gap_2, x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf15 = extern_kernels.convolution(buf14, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg19_1
        del buf14
        buf16 = empty_strided((8, 2, 1, 64), (128, 64, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf15, arg20_1, buf16, 1024, grid=grid(1024), stream=stream0)
        del arg20_1
        del buf15
        buf17 = buf8; del buf8  # reuse
        # Source Nodes: [mul, out_3, out_8], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_8.run(buf10, buf16, buf17, 2097152, grid=grid(2097152), stream=stream0)
        del buf10
        # Source Nodes: [mul, out_3, out_8], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf18 = extern_kernels.convolution(buf17, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg21_1
        del buf17
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_1], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf6, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg24_1
        del buf6
        buf20 = buf18; del buf18  # reuse
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [out_10, out_9, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf21, arg536_1, arg537_1, arg22_1, arg23_1, buf19, arg539_1, arg540_1, arg25_1, arg26_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg22_1
        del arg23_1
        del arg25_1
        del arg26_1
        del arg536_1
        del arg537_1
        del arg539_1
        del arg540_1
        del buf19
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg27_1
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [out_13, out_14, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf23, arg542_1, arg543_1, arg28_1, arg29_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg28_1
        del arg29_1
        del arg542_1
        del arg543_1
        # Source Nodes: [out_13, out_14, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf24, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg30_1
        buf25 = buf24; del buf24  # reuse
        # Source Nodes: [x_13, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf25, arg545_1, arg546_1, arg31_1, arg32_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg31_1
        del arg32_1
        del arg545_1
        del arg546_1
        buf26 = reinterpret_tensor(buf12, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf12  # reuse
        buf27 = reinterpret_tensor(buf26, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf26  # reuse
        # Source Nodes: [x_gap_5, x_gap_6, x_gap_7], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_red_fused_convolution_mean_sum_5.run(buf27, buf25, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap_5, x_gap_6, x_gap_7], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf28 = extern_kernels.convolution(buf27, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg33_1
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [x_attn_2, x_gap_5, x_gap_6, x_gap_7, x_gap_8, x_gap_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf29, arg34_1, arg548_1, arg549_1, arg35_1, arg36_1, 256, grid=grid(256), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        del arg548_1
        del arg549_1
        # Source Nodes: [x_attn_2, x_gap_5, x_gap_6, x_gap_7, x_gap_8, x_gap_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf30 = extern_kernels.convolution(buf29, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg37_1
        del buf29
        buf31 = buf16; del buf16  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf30, arg38_1, buf31, 1024, grid=grid(1024), stream=stream0)
        del arg38_1
        del buf30
        buf32 = buf23; del buf23  # reuse
        # Source Nodes: [mul_1, out_15, out_20], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_8.run(buf25, buf31, buf32, 2097152, grid=grid(2097152), stream=stream0)
        del buf25
        # Source Nodes: [mul_1, out_15, out_20], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf33 = extern_kernels.convolution(buf32, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg39_1
        del buf32
        buf34 = buf21; del buf21  # reuse
        # Source Nodes: [out_21, out_22, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf34, buf33, arg551_1, arg552_1, arg40_1, arg41_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg40_1
        del arg41_1
        del arg551_1
        del arg552_1
        del buf33
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg42_1
        buf36 = buf35; del buf35  # reuse
        # Source Nodes: [out_25, out_26, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf36, arg554_1, arg555_1, arg43_1, arg44_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        del arg44_1
        del arg554_1
        del arg555_1
        # Source Nodes: [out_25, out_26, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf37, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg45_1
        buf38 = buf37; del buf37  # reuse
        # Source Nodes: [x_21, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf38, arg557_1, arg558_1, arg46_1, arg47_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg46_1
        del arg47_1
        del arg557_1
        del arg558_1
        buf39 = reinterpret_tensor(buf27, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf27  # reuse
        buf40 = reinterpret_tensor(buf39, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf39  # reuse
        # Source Nodes: [x_gap_10, x_gap_11, x_gap_12], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_red_fused_convolution_mean_sum_5.run(buf40, buf38, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap_10, x_gap_11, x_gap_12], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf41 = extern_kernels.convolution(buf40, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg48_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Source Nodes: [x_attn_4, x_gap_10, x_gap_11, x_gap_12, x_gap_13, x_gap_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf42, arg49_1, arg560_1, arg561_1, arg50_1, arg51_1, 256, grid=grid(256), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        del arg560_1
        del arg561_1
        # Source Nodes: [x_attn_4, x_gap_10, x_gap_11, x_gap_12, x_gap_13, x_gap_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf43 = extern_kernels.convolution(buf42, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg52_1
        del buf42
        buf44 = buf31; del buf31  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf43, arg53_1, buf44, 1024, grid=grid(1024), stream=stream0)
        del arg53_1
        del buf43
        buf45 = buf36; del buf36  # reuse
        # Source Nodes: [mul_2, out_27, out_32], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_8.run(buf38, buf44, buf45, 2097152, grid=grid(2097152), stream=stream0)
        del buf38
        # Source Nodes: [mul_2, out_27, out_32], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf46 = extern_kernels.convolution(buf45, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg54_1
        buf47 = buf34; del buf34  # reuse
        # Source Nodes: [out_33, out_34, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf47, buf46, arg563_1, arg564_1, arg55_1, arg56_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg55_1
        del arg563_1
        del arg564_1
        del arg56_1
        del buf46
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg57_1
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [out_37, out_38, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf49, arg566_1, arg567_1, arg58_1, arg59_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg566_1
        del arg567_1
        del arg58_1
        del arg59_1
        # Source Nodes: [out_37, out_38, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf50 = extern_kernels.convolution(buf49, arg60_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf50, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg60_1
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [x_30, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf51, arg569_1, arg570_1, arg61_1, arg62_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg569_1
        del arg570_1
        del arg61_1
        del arg62_1
        buf52 = reinterpret_tensor(buf44, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf44  # reuse
        buf53 = reinterpret_tensor(buf52, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf52  # reuse
        # Source Nodes: [x_gap_15, x_gap_16, x_gap_17], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_red_fused_convolution_mean_sum_12.run(buf53, buf51, 1024, 4096, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_15, x_gap_16, x_gap_17], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf54 = extern_kernels.convolution(buf53, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg63_1
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [x_attn_6, x_gap_15, x_gap_16, x_gap_17, x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf55, arg64_1, arg572_1, arg573_1, arg65_1, arg66_1, 512, grid=grid(512), stream=stream0)
        del arg572_1
        del arg573_1
        del arg64_1
        del arg65_1
        del arg66_1
        # Source Nodes: [x_attn_6, x_gap_15, x_gap_16, x_gap_17, x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf56 = extern_kernels.convolution(buf55, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg67_1
        del buf55
        buf57 = empty_strided((8, 2, 1, 128), (256, 128, 2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf56, arg68_1, buf57, 2048, grid=grid(2048), stream=stream0)
        del arg68_1
        del buf56
        buf58 = buf49; del buf49  # reuse
        # Source Nodes: [mul_3, out_39], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_15.run(buf51, buf57, buf58, 4194304, grid=grid(4194304), stream=stream0)
        del buf51
        buf59 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_3, out_39, out_44], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
        triton_poi_fused_avg_pool2d_mul_sum_16.run(buf58, buf59, 1048576, grid=grid(1048576), stream=stream0)
        del buf58
        # Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg69_1
        del buf59
        buf61 = reinterpret_tensor(buf45, (8, 256, 32, 32), (262144, 1024, 32, 1), 0); del buf45  # reuse
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0, getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_17.run(buf47, buf61, 2097152, grid=grid(2097152), stream=stream0)
        del buf47
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0, getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg72_1
        del buf61
        buf63 = buf60; del buf60  # reuse
        buf64 = buf63; del buf63  # reuse
        # Source Nodes: [out_46, out_47, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf64, arg575_1, arg576_1, arg70_1, arg71_1, buf62, arg578_1, arg579_1, arg73_1, arg74_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg575_1
        del arg576_1
        del arg578_1
        del arg579_1
        del arg70_1
        del arg71_1
        del arg73_1
        del arg74_1
        del buf62
        # Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg75_1
        buf66 = buf65; del buf65  # reuse
        # Source Nodes: [out_50, out_51, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf66, arg581_1, arg582_1, arg76_1, arg77_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg581_1
        del arg582_1
        del arg76_1
        del arg77_1
        # Source Nodes: [out_50, out_51, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf67 = extern_kernels.convolution(buf66, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf67, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg78_1
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [x_38, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf68, arg584_1, arg585_1, arg79_1, arg80_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg584_1
        del arg585_1
        del arg79_1
        del arg80_1
        buf69 = reinterpret_tensor(buf53, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf53  # reuse
        buf70 = reinterpret_tensor(buf69, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf69  # reuse
        # Source Nodes: [x_gap_20, x_gap_21, x_gap_22], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_21.run(buf70, buf68, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_20, x_gap_21, x_gap_22], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf71 = extern_kernels.convolution(buf70, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg81_1
        buf72 = buf71; del buf71  # reuse
        # Source Nodes: [x_attn_8, x_gap_20, x_gap_21, x_gap_22, x_gap_23, x_gap_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf72, arg82_1, arg587_1, arg588_1, arg83_1, arg84_1, 512, grid=grid(512), stream=stream0)
        del arg587_1
        del arg588_1
        del arg82_1
        del arg83_1
        del arg84_1
        # Source Nodes: [x_attn_8, x_gap_20, x_gap_21, x_gap_22, x_gap_23, x_gap_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf73 = extern_kernels.convolution(buf72, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg85_1
        del buf72
        buf74 = buf57; del buf57  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf73, arg86_1, buf74, 2048, grid=grid(2048), stream=stream0)
        del arg86_1
        del buf73
        buf75 = buf66; del buf66  # reuse
        # Source Nodes: [mul_4, out_52, out_57], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_22.run(buf68, buf74, buf75, 1048576, grid=grid(1048576), stream=stream0)
        del buf68
        # Source Nodes: [mul_4, out_52, out_57], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf76 = extern_kernels.convolution(buf75, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg87_1
        del buf75
        buf77 = buf64; del buf64  # reuse
        # Source Nodes: [out_58, out_59, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf77, buf76, arg590_1, arg591_1, arg88_1, arg89_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg590_1
        del arg591_1
        del arg88_1
        del arg89_1
        del buf76
        # Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg90_1
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [out_62, out_63, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf79, arg593_1, arg594_1, arg91_1, arg92_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg593_1
        del arg594_1
        del arg91_1
        del arg92_1
        # Source Nodes: [out_62, out_63, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf80, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg93_1
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_46, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf81, arg596_1, arg597_1, arg94_1, arg95_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg596_1
        del arg597_1
        del arg94_1
        del arg95_1
        buf82 = reinterpret_tensor(buf70, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf70  # reuse
        buf83 = reinterpret_tensor(buf82, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf82  # reuse
        # Source Nodes: [x_gap_25, x_gap_26, x_gap_27], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_21.run(buf83, buf81, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_25, x_gap_26, x_gap_27], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf84 = extern_kernels.convolution(buf83, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg96_1
        buf85 = buf84; del buf84  # reuse
        # Source Nodes: [x_attn_10, x_gap_25, x_gap_26, x_gap_27, x_gap_28, x_gap_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf85, arg97_1, arg599_1, arg600_1, arg98_1, arg99_1, 512, grid=grid(512), stream=stream0)
        del arg599_1
        del arg600_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Source Nodes: [x_attn_10, x_gap_25, x_gap_26, x_gap_27, x_gap_28, x_gap_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf86 = extern_kernels.convolution(buf85, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg100_1
        del buf85
        buf87 = buf74; del buf74  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf86, arg101_1, buf87, 2048, grid=grid(2048), stream=stream0)
        del arg101_1
        del buf86
        buf88 = buf79; del buf79  # reuse
        # Source Nodes: [mul_5, out_64, out_69], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_22.run(buf81, buf87, buf88, 1048576, grid=grid(1048576), stream=stream0)
        del buf81
        # Source Nodes: [mul_5, out_64, out_69], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf89 = extern_kernels.convolution(buf88, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg102_1
        del buf88
        buf90 = buf77; del buf77  # reuse
        # Source Nodes: [out_70, out_71, shortcut_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf90, buf89, arg602_1, arg603_1, arg103_1, arg104_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg103_1
        del arg104_1
        del arg602_1
        del arg603_1
        del buf89
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg105_1
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [out_74, out_75, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf92, arg605_1, arg606_1, arg106_1, arg107_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg106_1
        del arg107_1
        del arg605_1
        del arg606_1
        # Source Nodes: [out_74, out_75, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf93, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg108_1
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [x_54, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf94, arg608_1, arg609_1, arg109_1, arg110_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg109_1
        del arg110_1
        del arg608_1
        del arg609_1
        buf95 = reinterpret_tensor(buf83, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf83  # reuse
        buf96 = reinterpret_tensor(buf95, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf95  # reuse
        # Source Nodes: [x_gap_30, x_gap_31, x_gap_32], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_21.run(buf96, buf94, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_30, x_gap_31, x_gap_32], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf97 = extern_kernels.convolution(buf96, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg111_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Source Nodes: [x_attn_12, x_gap_30, x_gap_31, x_gap_32, x_gap_33, x_gap_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf98, arg112_1, arg611_1, arg612_1, arg113_1, arg114_1, 512, grid=grid(512), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg611_1
        del arg612_1
        # Source Nodes: [x_attn_12, x_gap_30, x_gap_31, x_gap_32, x_gap_33, x_gap_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf99 = extern_kernels.convolution(buf98, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg115_1
        del buf98
        buf100 = buf87; del buf87  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf99, arg116_1, buf100, 2048, grid=grid(2048), stream=stream0)
        del arg116_1
        del buf99
        buf101 = buf92; del buf92  # reuse
        # Source Nodes: [mul_6, out_76, out_81], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_22.run(buf94, buf100, buf101, 1048576, grid=grid(1048576), stream=stream0)
        del buf94
        # Source Nodes: [mul_6, out_76, out_81], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf102 = extern_kernels.convolution(buf101, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg117_1
        buf103 = buf102; del buf102  # reuse
        # Source Nodes: [out_82, out_83, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf103, arg614_1, arg615_1, arg118_1, arg119_1, buf90, 4194304, grid=grid(4194304), stream=stream0)
        del arg118_1
        del arg119_1
        del arg614_1
        del arg615_1
        del buf90
        # Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg120_1
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [out_86, out_87, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf105, arg617_1, arg618_1, arg121_1, arg122_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg121_1
        del arg122_1
        del arg617_1
        del arg618_1
        # Source Nodes: [out_86, out_87, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf106 = extern_kernels.convolution(buf105, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf106, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg123_1
        buf107 = buf106; del buf106  # reuse
        # Source Nodes: [x_63, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf107, arg620_1, arg621_1, arg124_1, arg125_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg124_1
        del arg125_1
        del arg620_1
        del arg621_1
        buf108 = reinterpret_tensor(buf100, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf100  # reuse
        buf109 = reinterpret_tensor(buf108, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf108  # reuse
        # Source Nodes: [x_gap_35, x_gap_36, x_gap_37], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_26.run(buf109, buf107, 2048, 1024, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_35, x_gap_36, x_gap_37], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf110 = extern_kernels.convolution(buf109, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg126_1
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [x_attn_14, x_gap_35, x_gap_36, x_gap_37, x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf111, arg127_1, arg623_1, arg624_1, arg128_1, arg129_1, 1024, grid=grid(1024), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg623_1
        del arg624_1
        # Source Nodes: [x_attn_14, x_gap_35, x_gap_36, x_gap_37, x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf112 = extern_kernels.convolution(buf111, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg130_1
        del buf111
        buf113 = empty_strided((8, 2, 1, 256), (512, 256, 4096, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf112, arg131_1, buf113, 4096, grid=grid(4096), stream=stream0)
        del arg131_1
        del buf112
        buf114 = buf105; del buf105  # reuse
        # Source Nodes: [mul_7, out_88], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_29.run(buf107, buf113, buf114, 2097152, grid=grid(2097152), stream=stream0)
        del buf107
        buf115 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_7, out_88, out_93], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
        triton_poi_fused_avg_pool2d_mul_sum_30.run(buf114, buf115, 524288, grid=grid(524288), stream=stream0)
        del buf114
        # Source Nodes: [out_94], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg132_1
        del buf115
        buf117 = reinterpret_tensor(buf101, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf101  # reuse
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0, getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_31.run(buf103, buf117, 1048576, grid=grid(1048576), stream=stream0)
        del buf103
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0, getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg135_1
        del buf117
        buf119 = buf116; del buf116  # reuse
        buf120 = buf119; del buf119  # reuse
        # Source Nodes: [out_95, out_96, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32.run(buf120, arg626_1, arg627_1, arg133_1, arg134_1, buf118, arg629_1, arg630_1, arg136_1, arg137_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg133_1
        del arg134_1
        del arg136_1
        del arg137_1
        del arg626_1
        del arg627_1
        del arg629_1
        del arg630_1
        del buf118
        # Source Nodes: [out_98], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg138_1
        buf122 = buf121; del buf121  # reuse
        # Source Nodes: [out_100, out_99, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf122, arg632_1, arg633_1, arg139_1, arg140_1, 524288, grid=grid(524288), stream=stream0)
        del arg139_1
        del arg140_1
        del arg632_1
        del arg633_1
        # Source Nodes: [out_100, out_99, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf123 = extern_kernels.convolution(buf122, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf123, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg141_1
        buf124 = buf123; del buf123  # reuse
        # Source Nodes: [x_71, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf124, arg635_1, arg636_1, arg142_1, arg143_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg142_1
        del arg143_1
        del arg635_1
        del arg636_1
        buf125 = reinterpret_tensor(buf109, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf109  # reuse
        buf126 = reinterpret_tensor(buf125, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf125  # reuse
        # Source Nodes: [x_gap_40, x_gap_41, x_gap_42], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf126, buf124, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_40, x_gap_41, x_gap_42], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf127 = extern_kernels.convolution(buf126, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg144_1
        buf128 = buf127; del buf127  # reuse
        # Source Nodes: [x_attn_16, x_gap_40, x_gap_41, x_gap_42, x_gap_43, x_gap_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf128, arg145_1, arg638_1, arg639_1, arg146_1, arg147_1, 1024, grid=grid(1024), stream=stream0)
        del arg145_1
        del arg146_1
        del arg147_1
        del arg638_1
        del arg639_1
        # Source Nodes: [x_attn_16, x_gap_40, x_gap_41, x_gap_42, x_gap_43, x_gap_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf129 = extern_kernels.convolution(buf128, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg148_1
        del buf128
        buf130 = buf113; del buf113  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf129, arg149_1, buf130, 4096, grid=grid(4096), stream=stream0)
        del arg149_1
        del buf129
        buf131 = buf122; del buf122  # reuse
        # Source Nodes: [mul_8, out_101, out_106], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf124, buf130, buf131, 524288, grid=grid(524288), stream=stream0)
        del buf124
        # Source Nodes: [mul_8, out_101, out_106], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf132 = extern_kernels.convolution(buf131, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg150_1
        del buf131
        buf133 = buf120; del buf120  # reuse
        # Source Nodes: [out_107, out_108, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf133, buf132, arg641_1, arg642_1, arg151_1, arg152_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg151_1
        del arg152_1
        del arg641_1
        del arg642_1
        del buf132
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg153_1
        buf135 = buf134; del buf134  # reuse
        # Source Nodes: [out_111, out_112, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf135, arg644_1, arg645_1, arg154_1, arg155_1, 524288, grid=grid(524288), stream=stream0)
        del arg154_1
        del arg155_1
        del arg644_1
        del arg645_1
        # Source Nodes: [out_111, out_112, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf136 = extern_kernels.convolution(buf135, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf136, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg156_1
        buf137 = buf136; del buf136  # reuse
        # Source Nodes: [x_79, x_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf137, arg647_1, arg648_1, arg157_1, arg158_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg157_1
        del arg158_1
        del arg647_1
        del arg648_1
        buf138 = reinterpret_tensor(buf126, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf126  # reuse
        buf139 = reinterpret_tensor(buf138, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf138  # reuse
        # Source Nodes: [x_gap_45, x_gap_46, x_gap_47], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf139, buf137, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_45, x_gap_46, x_gap_47], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf140 = extern_kernels.convolution(buf139, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg159_1
        buf141 = buf140; del buf140  # reuse
        # Source Nodes: [x_attn_18, x_gap_45, x_gap_46, x_gap_47, x_gap_48, x_gap_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf141, arg160_1, arg650_1, arg651_1, arg161_1, arg162_1, 1024, grid=grid(1024), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg650_1
        del arg651_1
        # Source Nodes: [x_attn_18, x_gap_45, x_gap_46, x_gap_47, x_gap_48, x_gap_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf142 = extern_kernels.convolution(buf141, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg163_1
        del buf141
        buf143 = buf130; del buf130  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf142, arg164_1, buf143, 4096, grid=grid(4096), stream=stream0)
        del arg164_1
        del buf142
        buf144 = buf135; del buf135  # reuse
        # Source Nodes: [mul_9, out_113, out_118], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf137, buf143, buf144, 524288, grid=grid(524288), stream=stream0)
        del buf137
        # Source Nodes: [mul_9, out_113, out_118], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf145 = extern_kernels.convolution(buf144, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg165_1
        del buf144
        buf146 = buf133; del buf133  # reuse
        # Source Nodes: [out_119, out_120, shortcut_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf146, buf145, arg653_1, arg654_1, arg166_1, arg167_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg166_1
        del arg167_1
        del arg653_1
        del arg654_1
        del buf145
        # Source Nodes: [out_122], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg168_1
        buf148 = buf147; del buf147  # reuse
        # Source Nodes: [out_123, out_124, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf148, arg656_1, arg657_1, arg169_1, arg170_1, 524288, grid=grid(524288), stream=stream0)
        del arg169_1
        del arg170_1
        del arg656_1
        del arg657_1
        # Source Nodes: [out_123, out_124, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf149 = extern_kernels.convolution(buf148, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf149, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg171_1
        buf150 = buf149; del buf149  # reuse
        # Source Nodes: [x_87, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf150, arg659_1, arg660_1, arg172_1, arg173_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg172_1
        del arg173_1
        del arg659_1
        del arg660_1
        buf151 = reinterpret_tensor(buf139, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf139  # reuse
        buf152 = reinterpret_tensor(buf151, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf151  # reuse
        # Source Nodes: [x_gap_50, x_gap_51, x_gap_52], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf152, buf150, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_50, x_gap_51, x_gap_52], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf153 = extern_kernels.convolution(buf152, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg174_1
        buf154 = buf153; del buf153  # reuse
        # Source Nodes: [x_attn_20, x_gap_50, x_gap_51, x_gap_52, x_gap_53, x_gap_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf154, arg175_1, arg662_1, arg663_1, arg176_1, arg177_1, 1024, grid=grid(1024), stream=stream0)
        del arg175_1
        del arg176_1
        del arg177_1
        del arg662_1
        del arg663_1
        # Source Nodes: [x_attn_20, x_gap_50, x_gap_51, x_gap_52, x_gap_53, x_gap_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf155 = extern_kernels.convolution(buf154, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg178_1
        del buf154
        buf156 = buf143; del buf143  # reuse
        # Source Nodes: [x_92], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf155, arg179_1, buf156, 4096, grid=grid(4096), stream=stream0)
        del arg179_1
        del buf155
        buf157 = buf148; del buf148  # reuse
        # Source Nodes: [mul_10, out_125, out_130], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf150, buf156, buf157, 524288, grid=grid(524288), stream=stream0)
        del buf150
        # Source Nodes: [mul_10, out_125, out_130], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf158 = extern_kernels.convolution(buf157, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg180_1
        del buf157
        buf159 = buf146; del buf146  # reuse
        # Source Nodes: [out_131, out_132, shortcut_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf159, buf158, arg665_1, arg666_1, arg181_1, arg182_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg181_1
        del arg182_1
        del arg665_1
        del arg666_1
        del buf158
        # Source Nodes: [out_134], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg183_1
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [out_135, out_136, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf161, arg668_1, arg669_1, arg184_1, arg185_1, 524288, grid=grid(524288), stream=stream0)
        del arg184_1
        del arg185_1
        del arg668_1
        del arg669_1
        # Source Nodes: [out_135, out_136, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf162 = extern_kernels.convolution(buf161, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf162, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg186_1
        buf163 = buf162; del buf162  # reuse
        # Source Nodes: [x_95, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf163, arg671_1, arg672_1, arg187_1, arg188_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg187_1
        del arg188_1
        del arg671_1
        del arg672_1
        buf164 = reinterpret_tensor(buf152, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf152  # reuse
        buf165 = reinterpret_tensor(buf164, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf164  # reuse
        # Source Nodes: [x_gap_55, x_gap_56, x_gap_57], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf165, buf163, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_55, x_gap_56, x_gap_57], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf166 = extern_kernels.convolution(buf165, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg189_1
        buf167 = buf166; del buf166  # reuse
        # Source Nodes: [x_attn_22, x_gap_55, x_gap_56, x_gap_57, x_gap_58, x_gap_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf167, arg190_1, arg674_1, arg675_1, arg191_1, arg192_1, 1024, grid=grid(1024), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        del arg674_1
        del arg675_1
        # Source Nodes: [x_attn_22, x_gap_55, x_gap_56, x_gap_57, x_gap_58, x_gap_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf168 = extern_kernels.convolution(buf167, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg193_1
        del buf167
        buf169 = buf156; del buf156  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf168, arg194_1, buf169, 4096, grid=grid(4096), stream=stream0)
        del arg194_1
        del buf168
        buf170 = buf161; del buf161  # reuse
        # Source Nodes: [mul_11, out_137, out_142], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf163, buf169, buf170, 524288, grid=grid(524288), stream=stream0)
        del buf163
        # Source Nodes: [mul_11, out_137, out_142], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf171 = extern_kernels.convolution(buf170, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg195_1
        del buf170
        buf172 = buf159; del buf159  # reuse
        # Source Nodes: [out_143, out_144, shortcut_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf172, buf171, arg677_1, arg678_1, arg196_1, arg197_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg196_1
        del arg197_1
        del arg677_1
        del arg678_1
        del buf171
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg198_1
        buf174 = buf173; del buf173  # reuse
        # Source Nodes: [out_147, out_148, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf174, arg680_1, arg681_1, arg199_1, arg200_1, 524288, grid=grid(524288), stream=stream0)
        del arg199_1
        del arg200_1
        del arg680_1
        del arg681_1
        # Source Nodes: [out_147, out_148, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf175 = extern_kernels.convolution(buf174, arg201_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf175, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg201_1
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [x_103, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf176, arg683_1, arg684_1, arg202_1, arg203_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg202_1
        del arg203_1
        del arg683_1
        del arg684_1
        buf177 = reinterpret_tensor(buf165, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf165  # reuse
        buf178 = reinterpret_tensor(buf177, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf177  # reuse
        # Source Nodes: [x_gap_60, x_gap_61, x_gap_62], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf178, buf176, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_60, x_gap_61, x_gap_62], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf179 = extern_kernels.convolution(buf178, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg204_1
        buf180 = buf179; del buf179  # reuse
        # Source Nodes: [x_attn_24, x_gap_60, x_gap_61, x_gap_62, x_gap_63, x_gap_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf180, arg205_1, arg686_1, arg687_1, arg206_1, arg207_1, 1024, grid=grid(1024), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        del arg686_1
        del arg687_1
        # Source Nodes: [x_attn_24, x_gap_60, x_gap_61, x_gap_62, x_gap_63, x_gap_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf181 = extern_kernels.convolution(buf180, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg208_1
        del buf180
        buf182 = buf169; del buf169  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf181, arg209_1, buf182, 4096, grid=grid(4096), stream=stream0)
        del arg209_1
        del buf181
        buf183 = buf174; del buf174  # reuse
        # Source Nodes: [mul_12, out_149, out_154], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf176, buf182, buf183, 524288, grid=grid(524288), stream=stream0)
        del buf176
        # Source Nodes: [mul_12, out_149, out_154], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf184 = extern_kernels.convolution(buf183, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg210_1
        del buf183
        buf185 = buf172; del buf172  # reuse
        # Source Nodes: [out_155, out_156, shortcut_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf185, buf184, arg689_1, arg690_1, arg211_1, arg212_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg211_1
        del arg212_1
        del arg689_1
        del arg690_1
        del buf184
        # Source Nodes: [out_158], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg213_1
        buf187 = buf186; del buf186  # reuse
        # Source Nodes: [out_159, out_160, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf187, arg692_1, arg693_1, arg214_1, arg215_1, 524288, grid=grid(524288), stream=stream0)
        del arg214_1
        del arg215_1
        del arg692_1
        del arg693_1
        # Source Nodes: [out_159, out_160, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf188 = extern_kernels.convolution(buf187, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf188, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg216_1
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [x_111, x_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf189, arg695_1, arg696_1, arg217_1, arg218_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg217_1
        del arg218_1
        del arg695_1
        del arg696_1
        buf190 = reinterpret_tensor(buf178, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf178  # reuse
        buf191 = reinterpret_tensor(buf190, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf190  # reuse
        # Source Nodes: [x_gap_65, x_gap_66, x_gap_67], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf191, buf189, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_65, x_gap_66, x_gap_67], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf192 = extern_kernels.convolution(buf191, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg219_1
        buf193 = buf192; del buf192  # reuse
        # Source Nodes: [x_attn_26, x_gap_65, x_gap_66, x_gap_67, x_gap_68, x_gap_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf193, arg220_1, arg698_1, arg699_1, arg221_1, arg222_1, 1024, grid=grid(1024), stream=stream0)
        del arg220_1
        del arg221_1
        del arg222_1
        del arg698_1
        del arg699_1
        # Source Nodes: [x_attn_26, x_gap_65, x_gap_66, x_gap_67, x_gap_68, x_gap_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf194 = extern_kernels.convolution(buf193, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg223_1
        del buf193
        buf195 = buf182; del buf182  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf194, arg224_1, buf195, 4096, grid=grid(4096), stream=stream0)
        del arg224_1
        del buf194
        buf196 = buf187; del buf187  # reuse
        # Source Nodes: [mul_13, out_161, out_166], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf189, buf195, buf196, 524288, grid=grid(524288), stream=stream0)
        del buf189
        # Source Nodes: [mul_13, out_161, out_166], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf197 = extern_kernels.convolution(buf196, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg225_1
        del buf196
        buf198 = buf185; del buf185  # reuse
        # Source Nodes: [out_167, out_168, shortcut_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf198, buf197, arg701_1, arg702_1, arg226_1, arg227_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg226_1
        del arg227_1
        del arg701_1
        del arg702_1
        del buf197
        # Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg228_1
        buf200 = buf199; del buf199  # reuse
        # Source Nodes: [out_171, out_172, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf200, arg704_1, arg705_1, arg229_1, arg230_1, 524288, grid=grid(524288), stream=stream0)
        del arg229_1
        del arg230_1
        del arg704_1
        del arg705_1
        # Source Nodes: [out_171, out_172, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf201 = extern_kernels.convolution(buf200, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf201, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg231_1
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [x_119, x_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf202, arg707_1, arg708_1, arg232_1, arg233_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg232_1
        del arg233_1
        del arg707_1
        del arg708_1
        buf203 = reinterpret_tensor(buf191, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf191  # reuse
        buf204 = reinterpret_tensor(buf203, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf203  # reuse
        # Source Nodes: [x_gap_70, x_gap_71, x_gap_72], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf204, buf202, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_70, x_gap_71, x_gap_72], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf205 = extern_kernels.convolution(buf204, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg234_1
        buf206 = buf205; del buf205  # reuse
        # Source Nodes: [x_attn_28, x_gap_70, x_gap_71, x_gap_72, x_gap_73, x_gap_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf206, arg235_1, arg710_1, arg711_1, arg236_1, arg237_1, 1024, grid=grid(1024), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        del arg710_1
        del arg711_1
        # Source Nodes: [x_attn_28, x_gap_70, x_gap_71, x_gap_72, x_gap_73, x_gap_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf207 = extern_kernels.convolution(buf206, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg238_1
        del buf206
        buf208 = buf195; del buf195  # reuse
        # Source Nodes: [x_124], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf207, arg239_1, buf208, 4096, grid=grid(4096), stream=stream0)
        del arg239_1
        del buf207
        buf209 = buf200; del buf200  # reuse
        # Source Nodes: [mul_14, out_173, out_178], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf202, buf208, buf209, 524288, grid=grid(524288), stream=stream0)
        del buf202
        # Source Nodes: [mul_14, out_173, out_178], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf210 = extern_kernels.convolution(buf209, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg240_1
        del buf209
        buf211 = buf198; del buf198  # reuse
        # Source Nodes: [out_179, out_180, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf211, buf210, arg713_1, arg714_1, arg241_1, arg242_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg241_1
        del arg242_1
        del arg713_1
        del arg714_1
        del buf210
        # Source Nodes: [out_182], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg243_1
        buf213 = buf212; del buf212  # reuse
        # Source Nodes: [out_183, out_184, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf213, arg716_1, arg717_1, arg244_1, arg245_1, 524288, grid=grid(524288), stream=stream0)
        del arg244_1
        del arg245_1
        del arg716_1
        del arg717_1
        # Source Nodes: [out_183, out_184, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf214 = extern_kernels.convolution(buf213, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf214, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg246_1
        buf215 = buf214; del buf214  # reuse
        # Source Nodes: [x_127, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf215, arg719_1, arg720_1, arg247_1, arg248_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg247_1
        del arg248_1
        del arg719_1
        del arg720_1
        buf216 = reinterpret_tensor(buf204, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf204  # reuse
        buf217 = reinterpret_tensor(buf216, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf216  # reuse
        # Source Nodes: [x_gap_75, x_gap_76, x_gap_77], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf217, buf215, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_75, x_gap_76, x_gap_77], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf218 = extern_kernels.convolution(buf217, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg249_1
        buf219 = buf218; del buf218  # reuse
        # Source Nodes: [x_attn_30, x_gap_75, x_gap_76, x_gap_77, x_gap_78, x_gap_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf219, arg250_1, arg722_1, arg723_1, arg251_1, arg252_1, 1024, grid=grid(1024), stream=stream0)
        del arg250_1
        del arg251_1
        del arg252_1
        del arg722_1
        del arg723_1
        # Source Nodes: [x_attn_30, x_gap_75, x_gap_76, x_gap_77, x_gap_78, x_gap_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf220 = extern_kernels.convolution(buf219, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg253_1
        del buf219
        buf221 = buf208; del buf208  # reuse
        # Source Nodes: [x_132], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf220, arg254_1, buf221, 4096, grid=grid(4096), stream=stream0)
        del arg254_1
        del buf220
        buf222 = buf213; del buf213  # reuse
        # Source Nodes: [mul_15, out_185, out_190], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf215, buf221, buf222, 524288, grid=grid(524288), stream=stream0)
        del buf215
        # Source Nodes: [mul_15, out_185, out_190], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf223 = extern_kernels.convolution(buf222, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg255_1
        del buf222
        buf224 = buf211; del buf211  # reuse
        # Source Nodes: [out_191, out_192, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf224, buf223, arg725_1, arg726_1, arg256_1, arg257_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg256_1
        del arg257_1
        del arg725_1
        del arg726_1
        del buf223
        # Source Nodes: [out_194], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg258_1
        buf226 = buf225; del buf225  # reuse
        # Source Nodes: [out_195, out_196, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf226, arg728_1, arg729_1, arg259_1, arg260_1, 524288, grid=grid(524288), stream=stream0)
        del arg259_1
        del arg260_1
        del arg728_1
        del arg729_1
        # Source Nodes: [out_195, out_196, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf227 = extern_kernels.convolution(buf226, arg261_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf227, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg261_1
        buf228 = buf227; del buf227  # reuse
        # Source Nodes: [x_135, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf228, arg731_1, arg732_1, arg262_1, arg263_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg262_1
        del arg263_1
        del arg731_1
        del arg732_1
        buf229 = reinterpret_tensor(buf217, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf217  # reuse
        buf230 = reinterpret_tensor(buf229, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf229  # reuse
        # Source Nodes: [x_gap_80, x_gap_81, x_gap_82], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf230, buf228, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_80, x_gap_81, x_gap_82], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf231 = extern_kernels.convolution(buf230, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg264_1
        buf232 = buf231; del buf231  # reuse
        # Source Nodes: [x_attn_32, x_gap_80, x_gap_81, x_gap_82, x_gap_83, x_gap_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf232, arg265_1, arg734_1, arg735_1, arg266_1, arg267_1, 1024, grid=grid(1024), stream=stream0)
        del arg265_1
        del arg266_1
        del arg267_1
        del arg734_1
        del arg735_1
        # Source Nodes: [x_attn_32, x_gap_80, x_gap_81, x_gap_82, x_gap_83, x_gap_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf233 = extern_kernels.convolution(buf232, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg268_1
        del buf232
        buf234 = buf221; del buf221  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf233, arg269_1, buf234, 4096, grid=grid(4096), stream=stream0)
        del arg269_1
        del buf233
        buf235 = buf226; del buf226  # reuse
        # Source Nodes: [mul_16, out_197, out_202], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf228, buf234, buf235, 524288, grid=grid(524288), stream=stream0)
        del buf228
        # Source Nodes: [mul_16, out_197, out_202], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf236 = extern_kernels.convolution(buf235, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg270_1
        del buf235
        buf237 = buf224; del buf224  # reuse
        # Source Nodes: [out_203, out_204, shortcut_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf237, buf236, arg737_1, arg738_1, arg271_1, arg272_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg271_1
        del arg272_1
        del arg737_1
        del arg738_1
        del buf236
        # Source Nodes: [out_206], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg273_1
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [out_207, out_208, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf239, arg740_1, arg741_1, arg274_1, arg275_1, 524288, grid=grid(524288), stream=stream0)
        del arg274_1
        del arg275_1
        del arg740_1
        del arg741_1
        # Source Nodes: [out_207, out_208, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf240 = extern_kernels.convolution(buf239, arg276_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf240, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg276_1
        buf241 = buf240; del buf240  # reuse
        # Source Nodes: [x_143, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf241, arg743_1, arg744_1, arg277_1, arg278_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg277_1
        del arg278_1
        del arg743_1
        del arg744_1
        buf242 = reinterpret_tensor(buf230, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf230  # reuse
        buf243 = reinterpret_tensor(buf242, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf242  # reuse
        # Source Nodes: [x_gap_85, x_gap_86, x_gap_87], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf243, buf241, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_85, x_gap_86, x_gap_87], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf244 = extern_kernels.convolution(buf243, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg279_1
        buf245 = buf244; del buf244  # reuse
        # Source Nodes: [x_attn_34, x_gap_85, x_gap_86, x_gap_87, x_gap_88, x_gap_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf245, arg280_1, arg746_1, arg747_1, arg281_1, arg282_1, 1024, grid=grid(1024), stream=stream0)
        del arg280_1
        del arg281_1
        del arg282_1
        del arg746_1
        del arg747_1
        # Source Nodes: [x_attn_34, x_gap_85, x_gap_86, x_gap_87, x_gap_88, x_gap_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf246 = extern_kernels.convolution(buf245, arg283_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg283_1
        del buf245
        buf247 = buf234; del buf234  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf246, arg284_1, buf247, 4096, grid=grid(4096), stream=stream0)
        del arg284_1
        del buf246
        buf248 = buf239; del buf239  # reuse
        # Source Nodes: [mul_17, out_209, out_214], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf241, buf247, buf248, 524288, grid=grid(524288), stream=stream0)
        del buf241
        # Source Nodes: [mul_17, out_209, out_214], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf249 = extern_kernels.convolution(buf248, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg285_1
        del buf248
        buf250 = buf237; del buf237  # reuse
        # Source Nodes: [out_215, out_216, shortcut_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf250, buf249, arg749_1, arg750_1, arg286_1, arg287_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg286_1
        del arg287_1
        del arg749_1
        del arg750_1
        del buf249
        # Source Nodes: [out_218], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg288_1
        buf252 = buf251; del buf251  # reuse
        # Source Nodes: [out_219, out_220, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf252, arg752_1, arg753_1, arg289_1, arg290_1, 524288, grid=grid(524288), stream=stream0)
        del arg289_1
        del arg290_1
        del arg752_1
        del arg753_1
        # Source Nodes: [out_219, out_220, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf253 = extern_kernels.convolution(buf252, arg291_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf253, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg291_1
        buf254 = buf253; del buf253  # reuse
        # Source Nodes: [x_151, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf254, arg755_1, arg756_1, arg292_1, arg293_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg292_1
        del arg293_1
        del arg755_1
        del arg756_1
        buf255 = reinterpret_tensor(buf243, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf243  # reuse
        buf256 = reinterpret_tensor(buf255, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf255  # reuse
        # Source Nodes: [x_gap_90, x_gap_91, x_gap_92], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf256, buf254, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_90, x_gap_91, x_gap_92], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf257 = extern_kernels.convolution(buf256, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg294_1
        buf258 = buf257; del buf257  # reuse
        # Source Nodes: [x_attn_36, x_gap_90, x_gap_91, x_gap_92, x_gap_93, x_gap_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf258, arg295_1, arg758_1, arg759_1, arg296_1, arg297_1, 1024, grid=grid(1024), stream=stream0)
        del arg295_1
        del arg296_1
        del arg297_1
        del arg758_1
        del arg759_1
        # Source Nodes: [x_attn_36, x_gap_90, x_gap_91, x_gap_92, x_gap_93, x_gap_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf259 = extern_kernels.convolution(buf258, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg298_1
        del buf258
        buf260 = buf247; del buf247  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf259, arg299_1, buf260, 4096, grid=grid(4096), stream=stream0)
        del arg299_1
        del buf259
        buf261 = buf252; del buf252  # reuse
        # Source Nodes: [mul_18, out_221, out_226], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf254, buf260, buf261, 524288, grid=grid(524288), stream=stream0)
        del buf254
        # Source Nodes: [mul_18, out_221, out_226], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf262 = extern_kernels.convolution(buf261, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg300_1
        del buf261
        buf263 = buf250; del buf250  # reuse
        # Source Nodes: [out_227, out_228, shortcut_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf263, buf262, arg761_1, arg762_1, arg301_1, arg302_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg301_1
        del arg302_1
        del arg761_1
        del arg762_1
        del buf262
        # Source Nodes: [out_230], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg303_1
        buf265 = buf264; del buf264  # reuse
        # Source Nodes: [out_231, out_232, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf265, arg764_1, arg765_1, arg304_1, arg305_1, 524288, grid=grid(524288), stream=stream0)
        del arg304_1
        del arg305_1
        del arg764_1
        del arg765_1
        # Source Nodes: [out_231, out_232, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf266 = extern_kernels.convolution(buf265, arg306_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf266, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg306_1
        buf267 = buf266; del buf266  # reuse
        # Source Nodes: [x_159, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf267, arg767_1, arg768_1, arg307_1, arg308_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg307_1
        del arg308_1
        del arg767_1
        del arg768_1
        buf268 = reinterpret_tensor(buf256, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf256  # reuse
        buf269 = reinterpret_tensor(buf268, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf268  # reuse
        # Source Nodes: [x_gap_95, x_gap_96, x_gap_97], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf269, buf267, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_95, x_gap_96, x_gap_97], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf270 = extern_kernels.convolution(buf269, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg309_1
        buf271 = buf270; del buf270  # reuse
        # Source Nodes: [x_attn_38, x_gap_95, x_gap_96, x_gap_97, x_gap_98, x_gap_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf271, arg310_1, arg770_1, arg771_1, arg311_1, arg312_1, 1024, grid=grid(1024), stream=stream0)
        del arg310_1
        del arg311_1
        del arg312_1
        del arg770_1
        del arg771_1
        # Source Nodes: [x_attn_38, x_gap_95, x_gap_96, x_gap_97, x_gap_98, x_gap_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf272 = extern_kernels.convolution(buf271, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg313_1
        del buf271
        buf273 = buf260; del buf260  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf272, arg314_1, buf273, 4096, grid=grid(4096), stream=stream0)
        del arg314_1
        del buf272
        buf274 = buf265; del buf265  # reuse
        # Source Nodes: [mul_19, out_233, out_238], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf267, buf273, buf274, 524288, grid=grid(524288), stream=stream0)
        del buf267
        # Source Nodes: [mul_19, out_233, out_238], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf275 = extern_kernels.convolution(buf274, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg315_1
        del buf274
        buf276 = buf263; del buf263  # reuse
        # Source Nodes: [out_239, out_240, shortcut_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf276, buf275, arg773_1, arg774_1, arg316_1, arg317_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg316_1
        del arg317_1
        del arg773_1
        del arg774_1
        del buf275
        # Source Nodes: [out_242], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg318_1
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [out_243, out_244, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf278, arg776_1, arg777_1, arg319_1, arg320_1, 524288, grid=grid(524288), stream=stream0)
        del arg319_1
        del arg320_1
        del arg776_1
        del arg777_1
        # Source Nodes: [out_243, out_244, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf279 = extern_kernels.convolution(buf278, arg321_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf279, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg321_1
        buf280 = buf279; del buf279  # reuse
        # Source Nodes: [x_167, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf280, arg779_1, arg780_1, arg322_1, arg323_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg322_1
        del arg323_1
        del arg779_1
        del arg780_1
        buf281 = reinterpret_tensor(buf269, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf269  # reuse
        buf282 = reinterpret_tensor(buf281, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf281  # reuse
        # Source Nodes: [x_gap_100, x_gap_101, x_gap_102], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf282, buf280, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_100, x_gap_101, x_gap_102], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf283 = extern_kernels.convolution(buf282, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg324_1
        buf284 = buf283; del buf283  # reuse
        # Source Nodes: [x_attn_40, x_gap_100, x_gap_101, x_gap_102, x_gap_103, x_gap_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf284, arg325_1, arg782_1, arg783_1, arg326_1, arg327_1, 1024, grid=grid(1024), stream=stream0)
        del arg325_1
        del arg326_1
        del arg327_1
        del arg782_1
        del arg783_1
        # Source Nodes: [x_attn_40, x_gap_100, x_gap_101, x_gap_102, x_gap_103, x_gap_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf285 = extern_kernels.convolution(buf284, arg328_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg328_1
        del buf284
        buf286 = buf273; del buf273  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf285, arg329_1, buf286, 4096, grid=grid(4096), stream=stream0)
        del arg329_1
        del buf285
        buf287 = buf278; del buf278  # reuse
        # Source Nodes: [mul_20, out_245, out_250], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf280, buf286, buf287, 524288, grid=grid(524288), stream=stream0)
        del buf280
        # Source Nodes: [mul_20, out_245, out_250], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf288 = extern_kernels.convolution(buf287, arg330_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg330_1
        del buf287
        buf289 = buf276; del buf276  # reuse
        # Source Nodes: [out_251, out_252, shortcut_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf289, buf288, arg785_1, arg786_1, arg331_1, arg332_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg331_1
        del arg332_1
        del arg785_1
        del arg786_1
        del buf288
        # Source Nodes: [out_254], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg333_1
        buf291 = buf290; del buf290  # reuse
        # Source Nodes: [out_255, out_256, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf291, arg788_1, arg789_1, arg334_1, arg335_1, 524288, grid=grid(524288), stream=stream0)
        del arg334_1
        del arg335_1
        del arg788_1
        del arg789_1
        # Source Nodes: [out_255, out_256, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf292 = extern_kernels.convolution(buf291, arg336_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf292, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg336_1
        buf293 = buf292; del buf292  # reuse
        # Source Nodes: [x_175, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf293, arg791_1, arg792_1, arg337_1, arg338_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg337_1
        del arg338_1
        del arg791_1
        del arg792_1
        buf294 = reinterpret_tensor(buf282, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf282  # reuse
        buf295 = reinterpret_tensor(buf294, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf294  # reuse
        # Source Nodes: [x_gap_105, x_gap_106, x_gap_107], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf295, buf293, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_105, x_gap_106, x_gap_107], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf296 = extern_kernels.convolution(buf295, arg339_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg339_1
        buf297 = buf296; del buf296  # reuse
        # Source Nodes: [x_attn_42, x_gap_105, x_gap_106, x_gap_107, x_gap_108, x_gap_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf297, arg340_1, arg794_1, arg795_1, arg341_1, arg342_1, 1024, grid=grid(1024), stream=stream0)
        del arg340_1
        del arg341_1
        del arg342_1
        del arg794_1
        del arg795_1
        # Source Nodes: [x_attn_42, x_gap_105, x_gap_106, x_gap_107, x_gap_108, x_gap_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf298 = extern_kernels.convolution(buf297, arg343_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg343_1
        del buf297
        buf299 = buf286; del buf286  # reuse
        # Source Nodes: [x_180], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf298, arg344_1, buf299, 4096, grid=grid(4096), stream=stream0)
        del arg344_1
        del buf298
        buf300 = buf291; del buf291  # reuse
        # Source Nodes: [mul_21, out_257, out_262], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf293, buf299, buf300, 524288, grid=grid(524288), stream=stream0)
        del buf293
        # Source Nodes: [mul_21, out_257, out_262], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf301 = extern_kernels.convolution(buf300, arg345_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg345_1
        del buf300
        buf302 = buf289; del buf289  # reuse
        # Source Nodes: [out_263, out_264, shortcut_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf302, buf301, arg797_1, arg798_1, arg346_1, arg347_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg346_1
        del arg347_1
        del arg797_1
        del arg798_1
        del buf301
        # Source Nodes: [out_266], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, arg348_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg348_1
        buf304 = buf303; del buf303  # reuse
        # Source Nodes: [out_267, out_268, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf304, arg800_1, arg801_1, arg349_1, arg350_1, 524288, grid=grid(524288), stream=stream0)
        del arg349_1
        del arg350_1
        del arg800_1
        del arg801_1
        # Source Nodes: [out_267, out_268, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf305 = extern_kernels.convolution(buf304, arg351_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf305, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg351_1
        buf306 = buf305; del buf305  # reuse
        # Source Nodes: [x_183, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf306, arg803_1, arg804_1, arg352_1, arg353_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg352_1
        del arg353_1
        del arg803_1
        del arg804_1
        buf307 = reinterpret_tensor(buf295, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf295  # reuse
        buf308 = reinterpret_tensor(buf307, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf307  # reuse
        # Source Nodes: [x_gap_110, x_gap_111, x_gap_112], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf308, buf306, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_110, x_gap_111, x_gap_112], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf309 = extern_kernels.convolution(buf308, arg354_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg354_1
        buf310 = buf309; del buf309  # reuse
        # Source Nodes: [x_attn_44, x_gap_110, x_gap_111, x_gap_112, x_gap_113, x_gap_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf310, arg355_1, arg806_1, arg807_1, arg356_1, arg357_1, 1024, grid=grid(1024), stream=stream0)
        del arg355_1
        del arg356_1
        del arg357_1
        del arg806_1
        del arg807_1
        # Source Nodes: [x_attn_44, x_gap_110, x_gap_111, x_gap_112, x_gap_113, x_gap_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf311 = extern_kernels.convolution(buf310, arg358_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg358_1
        del buf310
        buf312 = buf299; del buf299  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf311, arg359_1, buf312, 4096, grid=grid(4096), stream=stream0)
        del arg359_1
        del buf311
        buf313 = buf304; del buf304  # reuse
        # Source Nodes: [mul_22, out_269, out_274], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf306, buf312, buf313, 524288, grid=grid(524288), stream=stream0)
        del buf306
        # Source Nodes: [mul_22, out_269, out_274], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf314 = extern_kernels.convolution(buf313, arg360_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg360_1
        del buf313
        buf315 = buf302; del buf302  # reuse
        # Source Nodes: [out_275, out_276, shortcut_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf315, buf314, arg809_1, arg810_1, arg361_1, arg362_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg361_1
        del arg362_1
        del arg809_1
        del arg810_1
        del buf314
        # Source Nodes: [out_278], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg363_1
        buf317 = buf316; del buf316  # reuse
        # Source Nodes: [out_279, out_280, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf317, arg812_1, arg813_1, arg364_1, arg365_1, 524288, grid=grid(524288), stream=stream0)
        del arg364_1
        del arg365_1
        del arg812_1
        del arg813_1
        # Source Nodes: [out_279, out_280, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf318 = extern_kernels.convolution(buf317, arg366_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf318, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg366_1
        buf319 = buf318; del buf318  # reuse
        # Source Nodes: [x_191, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf319, arg815_1, arg816_1, arg367_1, arg368_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg367_1
        del arg368_1
        del arg815_1
        del arg816_1
        buf320 = reinterpret_tensor(buf308, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf308  # reuse
        buf321 = reinterpret_tensor(buf320, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf320  # reuse
        # Source Nodes: [x_gap_115, x_gap_116, x_gap_117], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf321, buf319, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_115, x_gap_116, x_gap_117], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf322 = extern_kernels.convolution(buf321, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg369_1
        buf323 = buf322; del buf322  # reuse
        # Source Nodes: [x_attn_46, x_gap_115, x_gap_116, x_gap_117, x_gap_118, x_gap_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf323, arg370_1, arg818_1, arg819_1, arg371_1, arg372_1, 1024, grid=grid(1024), stream=stream0)
        del arg370_1
        del arg371_1
        del arg372_1
        del arg818_1
        del arg819_1
        # Source Nodes: [x_attn_46, x_gap_115, x_gap_116, x_gap_117, x_gap_118, x_gap_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf324 = extern_kernels.convolution(buf323, arg373_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg373_1
        del buf323
        buf325 = buf312; del buf312  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf324, arg374_1, buf325, 4096, grid=grid(4096), stream=stream0)
        del arg374_1
        del buf324
        buf326 = buf317; del buf317  # reuse
        # Source Nodes: [mul_23, out_281, out_286], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf319, buf325, buf326, 524288, grid=grid(524288), stream=stream0)
        del buf319
        # Source Nodes: [mul_23, out_281, out_286], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf327 = extern_kernels.convolution(buf326, arg375_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg375_1
        del buf326
        buf328 = buf315; del buf315  # reuse
        # Source Nodes: [out_287, out_288, shortcut_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf328, buf327, arg821_1, arg822_1, arg376_1, arg377_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg376_1
        del arg377_1
        del arg821_1
        del arg822_1
        del buf327
        # Source Nodes: [out_290], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, arg378_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg378_1
        buf330 = buf329; del buf329  # reuse
        # Source Nodes: [out_291, out_292, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf330, arg824_1, arg825_1, arg379_1, arg380_1, 524288, grid=grid(524288), stream=stream0)
        del arg379_1
        del arg380_1
        del arg824_1
        del arg825_1
        # Source Nodes: [out_291, out_292, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf331 = extern_kernels.convolution(buf330, arg381_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf331, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg381_1
        buf332 = buf331; del buf331  # reuse
        # Source Nodes: [x_199, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf332, arg827_1, arg828_1, arg382_1, arg383_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg382_1
        del arg383_1
        del arg827_1
        del arg828_1
        buf333 = reinterpret_tensor(buf321, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf321  # reuse
        buf334 = reinterpret_tensor(buf333, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf333  # reuse
        # Source Nodes: [x_gap_120, x_gap_121, x_gap_122], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf334, buf332, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_120, x_gap_121, x_gap_122], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf335 = extern_kernels.convolution(buf334, arg384_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg384_1
        buf336 = buf335; del buf335  # reuse
        # Source Nodes: [x_attn_48, x_gap_120, x_gap_121, x_gap_122, x_gap_123, x_gap_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf336, arg385_1, arg830_1, arg831_1, arg386_1, arg387_1, 1024, grid=grid(1024), stream=stream0)
        del arg385_1
        del arg386_1
        del arg387_1
        del arg830_1
        del arg831_1
        # Source Nodes: [x_attn_48, x_gap_120, x_gap_121, x_gap_122, x_gap_123, x_gap_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf337 = extern_kernels.convolution(buf336, arg388_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg388_1
        del buf336
        buf338 = buf325; del buf325  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf337, arg389_1, buf338, 4096, grid=grid(4096), stream=stream0)
        del arg389_1
        del buf337
        buf339 = buf330; del buf330  # reuse
        # Source Nodes: [mul_24, out_293, out_298], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf332, buf338, buf339, 524288, grid=grid(524288), stream=stream0)
        del buf332
        # Source Nodes: [mul_24, out_293, out_298], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf340 = extern_kernels.convolution(buf339, arg390_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg390_1
        del buf339
        buf341 = buf328; del buf328  # reuse
        # Source Nodes: [out_299, out_300, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf341, buf340, arg833_1, arg834_1, arg391_1, arg392_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg391_1
        del arg392_1
        del arg833_1
        del arg834_1
        del buf340
        # Source Nodes: [out_302], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, arg393_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg393_1
        buf343 = buf342; del buf342  # reuse
        # Source Nodes: [out_303, out_304, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf343, arg836_1, arg837_1, arg394_1, arg395_1, 524288, grid=grid(524288), stream=stream0)
        del arg394_1
        del arg395_1
        del arg836_1
        del arg837_1
        # Source Nodes: [out_303, out_304, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf344 = extern_kernels.convolution(buf343, arg396_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf344, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg396_1
        buf345 = buf344; del buf344  # reuse
        # Source Nodes: [x_207, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf345, arg839_1, arg840_1, arg397_1, arg398_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg397_1
        del arg398_1
        del arg839_1
        del arg840_1
        buf346 = reinterpret_tensor(buf334, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf334  # reuse
        buf347 = reinterpret_tensor(buf346, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf346  # reuse
        # Source Nodes: [x_gap_125, x_gap_126, x_gap_127], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf347, buf345, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_125, x_gap_126, x_gap_127], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf348 = extern_kernels.convolution(buf347, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg399_1
        buf349 = buf348; del buf348  # reuse
        # Source Nodes: [x_attn_50, x_gap_125, x_gap_126, x_gap_127, x_gap_128, x_gap_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf349, arg400_1, arg842_1, arg843_1, arg401_1, arg402_1, 1024, grid=grid(1024), stream=stream0)
        del arg400_1
        del arg401_1
        del arg402_1
        del arg842_1
        del arg843_1
        # Source Nodes: [x_attn_50, x_gap_125, x_gap_126, x_gap_127, x_gap_128, x_gap_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf350 = extern_kernels.convolution(buf349, arg403_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg403_1
        del buf349
        buf351 = buf338; del buf338  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf350, arg404_1, buf351, 4096, grid=grid(4096), stream=stream0)
        del arg404_1
        del buf350
        buf352 = buf343; del buf343  # reuse
        # Source Nodes: [mul_25, out_305, out_310], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf345, buf351, buf352, 524288, grid=grid(524288), stream=stream0)
        del buf345
        # Source Nodes: [mul_25, out_305, out_310], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf353 = extern_kernels.convolution(buf352, arg405_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg405_1
        del buf352
        buf354 = buf341; del buf341  # reuse
        # Source Nodes: [out_311, out_312, shortcut_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf354, buf353, arg845_1, arg846_1, arg406_1, arg407_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg406_1
        del arg407_1
        del arg845_1
        del arg846_1
        del buf353
        # Source Nodes: [out_314], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, arg408_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg408_1
        buf356 = buf355; del buf355  # reuse
        # Source Nodes: [out_315, out_316, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf356, arg848_1, arg849_1, arg409_1, arg410_1, 524288, grid=grid(524288), stream=stream0)
        del arg409_1
        del arg410_1
        del arg848_1
        del arg849_1
        # Source Nodes: [out_315, out_316, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf357 = extern_kernels.convolution(buf356, arg411_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf357, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg411_1
        buf358 = buf357; del buf357  # reuse
        # Source Nodes: [x_215, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf358, arg851_1, arg852_1, arg412_1, arg413_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg412_1
        del arg413_1
        del arg851_1
        del arg852_1
        buf359 = reinterpret_tensor(buf347, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf347  # reuse
        buf360 = reinterpret_tensor(buf359, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf359  # reuse
        # Source Nodes: [x_gap_130, x_gap_131, x_gap_132], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf360, buf358, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_130, x_gap_131, x_gap_132], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf361 = extern_kernels.convolution(buf360, arg414_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg414_1
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [x_attn_52, x_gap_130, x_gap_131, x_gap_132, x_gap_133, x_gap_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf362, arg415_1, arg854_1, arg855_1, arg416_1, arg417_1, 1024, grid=grid(1024), stream=stream0)
        del arg415_1
        del arg416_1
        del arg417_1
        del arg854_1
        del arg855_1
        # Source Nodes: [x_attn_52, x_gap_130, x_gap_131, x_gap_132, x_gap_133, x_gap_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf363 = extern_kernels.convolution(buf362, arg418_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg418_1
        del buf362
        buf364 = buf351; del buf351  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf363, arg419_1, buf364, 4096, grid=grid(4096), stream=stream0)
        del arg419_1
        del buf363
        buf365 = buf356; del buf356  # reuse
        # Source Nodes: [mul_26, out_317, out_322], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf358, buf364, buf365, 524288, grid=grid(524288), stream=stream0)
        del buf358
        # Source Nodes: [mul_26, out_317, out_322], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf366 = extern_kernels.convolution(buf365, arg420_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg420_1
        del buf365
        buf367 = buf354; del buf354  # reuse
        # Source Nodes: [out_323, out_324, shortcut_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf367, buf366, arg857_1, arg858_1, arg421_1, arg422_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg421_1
        del arg422_1
        del arg857_1
        del arg858_1
        del buf366
        # Source Nodes: [out_326], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, arg423_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg423_1
        buf369 = buf368; del buf368  # reuse
        # Source Nodes: [out_327, out_328, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf369, arg860_1, arg861_1, arg424_1, arg425_1, 524288, grid=grid(524288), stream=stream0)
        del arg424_1
        del arg425_1
        del arg860_1
        del arg861_1
        # Source Nodes: [out_327, out_328, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf370 = extern_kernels.convolution(buf369, arg426_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf370, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg426_1
        buf371 = buf370; del buf370  # reuse
        # Source Nodes: [x_223, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf371, arg863_1, arg864_1, arg427_1, arg428_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg427_1
        del arg428_1
        del arg863_1
        del arg864_1
        buf372 = reinterpret_tensor(buf360, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf360  # reuse
        buf373 = reinterpret_tensor(buf372, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf372  # reuse
        # Source Nodes: [x_gap_135, x_gap_136, x_gap_137], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf373, buf371, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_135, x_gap_136, x_gap_137], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf374 = extern_kernels.convolution(buf373, arg429_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg429_1
        buf375 = buf374; del buf374  # reuse
        # Source Nodes: [x_attn_54, x_gap_135, x_gap_136, x_gap_137, x_gap_138, x_gap_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf375, arg430_1, arg866_1, arg867_1, arg431_1, arg432_1, 1024, grid=grid(1024), stream=stream0)
        del arg430_1
        del arg431_1
        del arg432_1
        del arg866_1
        del arg867_1
        # Source Nodes: [x_attn_54, x_gap_135, x_gap_136, x_gap_137, x_gap_138, x_gap_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf376 = extern_kernels.convolution(buf375, arg433_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg433_1
        del buf375
        buf377 = buf364; del buf364  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf376, arg434_1, buf377, 4096, grid=grid(4096), stream=stream0)
        del arg434_1
        del buf376
        buf378 = buf369; del buf369  # reuse
        # Source Nodes: [mul_27, out_329, out_334], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf371, buf377, buf378, 524288, grid=grid(524288), stream=stream0)
        del buf371
        # Source Nodes: [mul_27, out_329, out_334], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf379 = extern_kernels.convolution(buf378, arg435_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg435_1
        del buf378
        buf380 = buf367; del buf367  # reuse
        # Source Nodes: [out_335, out_336, shortcut_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf380, buf379, arg869_1, arg870_1, arg436_1, arg437_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg436_1
        del arg437_1
        del arg869_1
        del arg870_1
        del buf379
        # Source Nodes: [out_338], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, arg438_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg438_1
        buf382 = buf381; del buf381  # reuse
        # Source Nodes: [out_339, out_340, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf382, arg872_1, arg873_1, arg439_1, arg440_1, 524288, grid=grid(524288), stream=stream0)
        del arg439_1
        del arg440_1
        del arg872_1
        del arg873_1
        # Source Nodes: [out_339, out_340, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf383 = extern_kernels.convolution(buf382, arg441_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf383, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg441_1
        buf384 = buf383; del buf383  # reuse
        # Source Nodes: [x_231, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf384, arg875_1, arg876_1, arg442_1, arg443_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg442_1
        del arg443_1
        del arg875_1
        del arg876_1
        buf385 = reinterpret_tensor(buf373, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf373  # reuse
        buf386 = reinterpret_tensor(buf385, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf385  # reuse
        # Source Nodes: [x_gap_140, x_gap_141, x_gap_142], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf386, buf384, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_140, x_gap_141, x_gap_142], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf387 = extern_kernels.convolution(buf386, arg444_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg444_1
        buf388 = buf387; del buf387  # reuse
        # Source Nodes: [x_attn_56, x_gap_140, x_gap_141, x_gap_142, x_gap_143, x_gap_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf388, arg445_1, arg878_1, arg879_1, arg446_1, arg447_1, 1024, grid=grid(1024), stream=stream0)
        del arg445_1
        del arg446_1
        del arg447_1
        del arg878_1
        del arg879_1
        # Source Nodes: [x_attn_56, x_gap_140, x_gap_141, x_gap_142, x_gap_143, x_gap_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf389 = extern_kernels.convolution(buf388, arg448_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg448_1
        del buf388
        buf390 = buf377; del buf377  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf389, arg449_1, buf390, 4096, grid=grid(4096), stream=stream0)
        del arg449_1
        del buf389
        buf391 = buf382; del buf382  # reuse
        # Source Nodes: [mul_28, out_341, out_346], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf384, buf390, buf391, 524288, grid=grid(524288), stream=stream0)
        del buf384
        # Source Nodes: [mul_28, out_341, out_346], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf392 = extern_kernels.convolution(buf391, arg450_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg450_1
        del buf391
        buf393 = buf380; del buf380  # reuse
        # Source Nodes: [out_347, out_348, shortcut_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf393, buf392, arg881_1, arg882_1, arg451_1, arg452_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg451_1
        del arg452_1
        del arg881_1
        del arg882_1
        del buf392
        # Source Nodes: [out_350], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, arg453_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg453_1
        buf395 = buf394; del buf394  # reuse
        # Source Nodes: [out_351, out_352, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf395, arg884_1, arg885_1, arg454_1, arg455_1, 524288, grid=grid(524288), stream=stream0)
        del arg454_1
        del arg455_1
        del arg884_1
        del arg885_1
        # Source Nodes: [out_351, out_352, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf396 = extern_kernels.convolution(buf395, arg456_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf396, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg456_1
        buf397 = buf396; del buf396  # reuse
        # Source Nodes: [x_239, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf397, arg887_1, arg888_1, arg457_1, arg458_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg457_1
        del arg458_1
        del arg887_1
        del arg888_1
        buf398 = reinterpret_tensor(buf386, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf386  # reuse
        buf399 = reinterpret_tensor(buf398, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf398  # reuse
        # Source Nodes: [x_gap_145, x_gap_146, x_gap_147], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_35.run(buf399, buf397, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_145, x_gap_146, x_gap_147], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf400 = extern_kernels.convolution(buf399, arg459_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg459_1
        del buf399
        buf401 = buf400; del buf400  # reuse
        # Source Nodes: [x_attn_58, x_gap_145, x_gap_146, x_gap_147, x_gap_148, x_gap_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf401, arg460_1, arg890_1, arg891_1, arg461_1, arg462_1, 1024, grid=grid(1024), stream=stream0)
        del arg460_1
        del arg461_1
        del arg462_1
        del arg890_1
        del arg891_1
        # Source Nodes: [x_attn_58, x_gap_145, x_gap_146, x_gap_147, x_gap_148, x_gap_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf402 = extern_kernels.convolution(buf401, arg463_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg463_1
        del buf401
        buf403 = buf390; del buf390  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf402, arg464_1, buf403, 4096, grid=grid(4096), stream=stream0)
        del arg464_1
        del buf402
        buf404 = buf395; del buf395  # reuse
        # Source Nodes: [mul_29, out_353, out_358], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_36.run(buf397, buf403, buf404, 524288, grid=grid(524288), stream=stream0)
        del buf397
        # Source Nodes: [mul_29, out_353, out_358], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf405 = extern_kernels.convolution(buf404, arg465_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg465_1
        buf406 = buf393; del buf393  # reuse
        # Source Nodes: [out_359, out_360, shortcut_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf406, buf405, arg893_1, arg894_1, arg466_1, arg467_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg466_1
        del arg467_1
        del arg893_1
        del arg894_1
        del buf405
        # Source Nodes: [out_362], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, arg468_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg468_1
        buf408 = buf407; del buf407  # reuse
        # Source Nodes: [out_363, out_364, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf408, arg896_1, arg897_1, arg469_1, arg470_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg469_1
        del arg470_1
        del arg896_1
        del arg897_1
        # Source Nodes: [out_363, out_364, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf409 = extern_kernels.convolution(buf408, arg471_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf409, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg471_1
        buf410 = buf409; del buf409  # reuse
        # Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf410, arg899_1, arg900_1, arg472_1, arg473_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg472_1
        del arg473_1
        del arg899_1
        del arg900_1
        buf411 = reinterpret_tensor(buf403, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf403  # reuse
        buf412 = reinterpret_tensor(buf411, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf411  # reuse
        # Source Nodes: [x_gap_150, x_gap_151, x_gap_152], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_39.run(buf412, buf410, 4096, 256, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_150, x_gap_151, x_gap_152], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf413 = extern_kernels.convolution(buf412, arg474_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg474_1
        buf414 = buf413; del buf413  # reuse
        # Source Nodes: [x_attn_60, x_gap_150, x_gap_151, x_gap_152, x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf414, arg475_1, arg902_1, arg903_1, arg476_1, arg477_1, 2048, grid=grid(2048), stream=stream0)
        del arg475_1
        del arg476_1
        del arg477_1
        del arg902_1
        del arg903_1
        # Source Nodes: [x_attn_60, x_gap_150, x_gap_151, x_gap_152, x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf415 = extern_kernels.convolution(buf414, arg478_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg478_1
        del buf414
        buf416 = empty_strided((8, 2, 1, 512), (1024, 512, 8192, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf415, arg479_1, buf416, 8192, grid=grid(8192), stream=stream0)
        del arg479_1
        del buf415
        buf417 = buf408; del buf408  # reuse
        # Source Nodes: [mul_30, out_365], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_42.run(buf410, buf416, buf417, 1048576, grid=grid(1048576), stream=stream0)
        del buf410
        buf418 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_30, out_365, out_370], Original ATen: [aten.avg_pool2d, aten.mul, aten.sum]
        triton_poi_fused_avg_pool2d_mul_sum_43.run(buf417, buf418, 262144, grid=grid(262144), stream=stream0)
        del buf417
        # Source Nodes: [out_371], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, arg480_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg480_1
        del buf418
        buf420 = reinterpret_tensor(buf404, (8, 1024, 8, 8), (65536, 64, 8, 1), 0); del buf404  # reuse
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0, getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_44.run(buf406, buf420, 524288, grid=grid(524288), stream=stream0)
        del buf406
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0, getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf421 = extern_kernels.convolution(buf420, arg483_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg483_1
        del buf420
        buf422 = buf419; del buf419  # reuse
        buf423 = buf422; del buf422  # reuse
        # Source Nodes: [out_372, out_373, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf423, arg905_1, arg906_1, arg481_1, arg482_1, buf421, arg908_1, arg909_1, arg484_1, arg485_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg481_1
        del arg482_1
        del arg484_1
        del arg485_1
        del arg905_1
        del arg906_1
        del arg908_1
        del arg909_1
        del buf421
        # Source Nodes: [out_375], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, arg486_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg486_1
        buf425 = buf424; del buf424  # reuse
        # Source Nodes: [out_376, out_377, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46.run(buf425, arg911_1, arg912_1, arg487_1, arg488_1, 262144, grid=grid(262144), stream=stream0)
        del arg487_1
        del arg488_1
        del arg911_1
        del arg912_1
        # Source Nodes: [out_376, out_377, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf426 = extern_kernels.convolution(buf425, arg489_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf426, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg489_1
        buf427 = buf426; del buf426  # reuse
        # Source Nodes: [x_256, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf427, arg914_1, arg915_1, arg490_1, arg491_1, 524288, grid=grid(524288), stream=stream0)
        del arg490_1
        del arg491_1
        del arg914_1
        del arg915_1
        buf428 = reinterpret_tensor(buf412, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf412  # reuse
        buf429 = reinterpret_tensor(buf428, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf428  # reuse
        # Source Nodes: [x_gap_155, x_gap_156, x_gap_157], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_48.run(buf429, buf427, 4096, 64, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_155, x_gap_156, x_gap_157], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf430 = extern_kernels.convolution(buf429, arg492_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg492_1
        buf431 = buf430; del buf430  # reuse
        # Source Nodes: [x_attn_62, x_gap_155, x_gap_156, x_gap_157, x_gap_158, x_gap_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf431, arg493_1, arg917_1, arg918_1, arg494_1, arg495_1, 2048, grid=grid(2048), stream=stream0)
        del arg493_1
        del arg494_1
        del arg495_1
        del arg917_1
        del arg918_1
        # Source Nodes: [x_attn_62, x_gap_155, x_gap_156, x_gap_157, x_gap_158, x_gap_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf432 = extern_kernels.convolution(buf431, arg496_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg496_1
        del buf431
        buf433 = buf416; del buf416  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf432, arg497_1, buf433, 8192, grid=grid(8192), stream=stream0)
        del arg497_1
        del buf432
        buf434 = buf425; del buf425  # reuse
        # Source Nodes: [mul_31, out_378, out_383], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_49.run(buf427, buf433, buf434, 262144, grid=grid(262144), stream=stream0)
        del buf427
        # Source Nodes: [mul_31, out_378, out_383], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf435 = extern_kernels.convolution(buf434, arg498_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg498_1
        del buf434
        buf436 = buf423; del buf423  # reuse
        # Source Nodes: [out_384, out_385, shortcut_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf436, buf435, arg920_1, arg921_1, arg499_1, arg500_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg499_1
        del arg500_1
        del arg920_1
        del arg921_1
        del buf435
        # Source Nodes: [out_387], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, arg501_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg501_1
        buf438 = buf437; del buf437  # reuse
        # Source Nodes: [out_388, out_389, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46.run(buf438, arg923_1, arg924_1, arg502_1, arg503_1, 262144, grid=grid(262144), stream=stream0)
        del arg502_1
        del arg503_1
        del arg923_1
        del arg924_1
        # Source Nodes: [out_388, out_389, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf439 = extern_kernels.convolution(buf438, arg504_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf439, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg504_1
        buf440 = buf439; del buf439  # reuse
        # Source Nodes: [x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf440, arg926_1, arg927_1, arg505_1, arg506_1, 524288, grid=grid(524288), stream=stream0)
        del arg505_1
        del arg506_1
        del arg926_1
        del arg927_1
        buf441 = reinterpret_tensor(buf429, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf429  # reuse
        buf442 = reinterpret_tensor(buf441, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf441  # reuse
        # Source Nodes: [x_gap_160, x_gap_161, x_gap_162], Original ATen: [aten.convolution, aten.mean, aten.sum]
        triton_per_fused_convolution_mean_sum_48.run(buf442, buf440, 4096, 64, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_160, x_gap_161, x_gap_162], Original ATen: [aten.convolution, aten.mean, aten.sum]
        buf443 = extern_kernels.convolution(buf442, arg507_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg507_1
        del buf442
        buf444 = buf443; del buf443  # reuse
        # Source Nodes: [x_attn_64, x_gap_160, x_gap_161, x_gap_162, x_gap_163, x_gap_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf444, arg508_1, arg929_1, arg930_1, arg509_1, arg510_1, 2048, grid=grid(2048), stream=stream0)
        del arg508_1
        del arg509_1
        del arg510_1
        del arg929_1
        del arg930_1
        # Source Nodes: [x_attn_64, x_gap_160, x_gap_161, x_gap_162, x_gap_163, x_gap_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.sum]
        buf445 = extern_kernels.convolution(buf444, arg511_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg511_1
        del buf444
        buf446 = buf433; del buf433  # reuse
        # Source Nodes: [x_269], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf445, arg512_1, buf446, 8192, grid=grid(8192), stream=stream0)
        del arg512_1
        del buf445
        buf447 = buf438; del buf438  # reuse
        # Source Nodes: [mul_32, out_390, out_395], Original ATen: [aten.convolution, aten.mul, aten.sum]
        triton_poi_fused_convolution_mul_sum_49.run(buf440, buf446, buf447, 262144, grid=grid(262144), stream=stream0)
        del buf440
        del buf446
        # Source Nodes: [mul_32, out_390, out_395], Original ATen: [aten.convolution, aten.mul, aten.sum]
        buf448 = extern_kernels.convolution(buf447, arg513_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg513_1
        del buf447
        buf449 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf450 = reinterpret_tensor(buf449, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf449  # reuse
        # Source Nodes: [out_396, out_397, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51.run(buf450, buf448, arg932_1, arg933_1, arg514_1, arg515_1, buf436, 16384, 64, grid=grid(16384), stream=stream0)
        del arg514_1
        del arg515_1
        del arg932_1
        del arg933_1
        del buf436
        del buf448
        buf451 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg517_1, reinterpret_tensor(buf450, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg516_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf451)
        del arg516_1
        del arg517_1
        return (buf451, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg521_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg524_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg527_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg530_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg533_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg536_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg539_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg542_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg545_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg548_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg551_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg554_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg557_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg560_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg563_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg566_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg569_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg572_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg575_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg578_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg581_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg584_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg587_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg590_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg593_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg596_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg599_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg602_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg605_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg608_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg611_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg614_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg617_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg620_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg623_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg626_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg629_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg632_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg635_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg638_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg641_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg644_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg647_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg650_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg653_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg656_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg659_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg662_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg665_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg668_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg671_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg674_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg677_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg680_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg683_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg686_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg689_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg692_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg695_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg698_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg701_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg704_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg707_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg710_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg713_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg716_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg719_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg722_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg725_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg728_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg731_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg734_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg737_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg740_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg743_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg746_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg749_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg752_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg755_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg758_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg761_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg764_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg767_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg770_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg773_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg776_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg779_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg782_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg785_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg788_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg791_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg794_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg797_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg800_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg803_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg806_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg809_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg812_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg815_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg818_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg821_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg824_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg827_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg830_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg833_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg836_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg839_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg842_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg845_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg848_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg851_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg854_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg857_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg860_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg863_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg866_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg869_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg872_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg875_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg878_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg881_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg884_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg887_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg890_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg893_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg896_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg899_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg902_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg905_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg908_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg911_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg914_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg917_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg920_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg923_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg926_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg929_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg932_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg935_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnest101e', benchmark_compiled_module)
