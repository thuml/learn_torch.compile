
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7ce4a3qwlvs5edoum3hzzikseixprdjgsw5wn2ly7u3mguijlcp.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
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


# kernel path: /tmp/torchinductor_youkaichao/yf/cyf43patlbahffht7zws3ecvadys44ofesccwq7wiki5xtpulnaw.py
# Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_11 => relu_1
# x_12 => convolution_2
# x_7 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjl2a7n2dlev3tulqjlx5mfqd2ol4gokyhm5cnkyfqurospkzqz.py
# Source Nodes: [x_13, x_17, x_se, x_se_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_13 => add_5, mul_7, mul_8, sub_2
# x_17 => relu_2
# x_se => mean
# x_se_1 => convolution_3
triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 24
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp15, rmask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = 3136.0
    tmp20 = tmp17 / tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca73ft6cgpso4s4m46nf7rcwt6s7zjkcxzo6khbwxlot2wqvqi3a.py
# Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se => mean
# x_se_1 => convolution_3
# x_se_2 => relu_3
# x_se_3 => convolution_4
triton_poi_fused_convolution_mean_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cyt2ysbkr5fgomp7zo3qlittjdroiilvc4urozwxyugh7q6fjtdg.py
# Source Nodes: [sigmoid, x_18, x_19, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid => sigmoid
# x_18 => mul_9
# x_19 => convolution_5
# x_se => mean
# x_se_1 => convolution_3
# x_se_2 => relu_3
# x_se_3 => convolution_4
triton_poi_fused_convolution_mean_mul_relu_sigmoid_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmm7pnrii3isdoqdigdwwjra4mkrqzxaqfq73gfdexsqftqquzj3.py
# Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_1 => relu_4
# x_20 => add_7, mul_11, mul_12, sub_3
# x_26 => add_9, mul_14, mul_15, sub_4
# x_30 => add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhxgtseladatlbtpkpsvubh6n6iapcrgz45i7vtoijwov5oxkhd.py
# Source Nodes: [x_35, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_35 => add_12, mul_17, mul_18, sub_5
# x_39 => relu_5
# x_40 => convolution_8
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 56
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


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2requllly2fspeo4jruqlpfi3o7mg6662bopdh7isqofxevlwz.py
# Source Nodes: [x_41, x_45, x_se_4, x_se_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_41 => add_14, mul_20, mul_21, sub_6
# x_45 => relu_6
# x_se_4 => mean_1
# x_se_5 => convolution_9
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 448
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
    x0 = xindex % 56
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxseyzyifr6om6xn7qz6lqk6s2hklyrqkq5pqulbo4pecospnr7y.py
# Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_4 => mean_1
# x_se_5 => convolution_9
# x_se_6 => relu_7
# x_se_7 => convolution_10
triton_poi_fused_convolution_mean_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coy736r4hyli6riubksy2yvnpwnvynnulb5qh4hoarvpuxjbvmek.py
# Source Nodes: [sigmoid_1, x_46, x_47, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_1 => sigmoid_1
# x_46 => mul_22
# x_47 => convolution_11
# x_se_4 => mean_1
# x_se_5 => convolution_9
# x_se_6 => relu_7
# x_se_7 => convolution_10
triton_poi_fused_convolution_mean_mul_relu_sigmoid_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5hciauulsmxd3cpsceerbfjwbxnqbl5cuj3wk3gqmv43yfvzh5.py
# Source Nodes: [shortcut_2, x_48, x_54, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_2 => relu_8
# x_48 => add_16, mul_24, mul_25, sub_7
# x_54 => add_18, mul_27, mul_28, sub_8
# x_58 => add_19
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccu5gtcyr7yk4wry3g3rqyquxfdm3l55plva6rfmmuxmsibascsp.py
# Source Nodes: [x_63, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_63 => add_21, mul_30, mul_31, sub_9
# x_67 => relu_9
# x_68 => convolution_14
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 152
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rpvw3gf5k54mq3jlpltwmyv75ra3gulkvfgub65hjbarrawcju.py
# Source Nodes: [x_69, x_73, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_69 => add_23, mul_33, mul_34, sub_10
# x_73 => relu_10
# x_se_8 => mean_2
# x_se_9 => convolution_15
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 152
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 196.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cryl3iw6e2fi4hthovzm6yp7v2dieav6uabdmrqcqref6iut5evo.py
# Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_10 => relu_11
# x_se_11 => convolution_16
# x_se_8 => mean_2
# x_se_9 => convolution_15
triton_poi_fused_convolution_mean_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugrjmnpplxylzimtj5arc4c2bcqmyykcvnx3g7lakp5dpw3dnya.py
# Source Nodes: [sigmoid_2, x_74, x_75, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_2 => sigmoid_2
# x_74 => mul_35
# x_75 => convolution_17
# x_se_10 => relu_11
# x_se_11 => convolution_16
# x_se_8 => mean_2
# x_se_9 => convolution_15
triton_poi_fused_convolution_mean_mul_relu_sigmoid_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcn2teern4fl54ic22zknuropx3dy4v25bkdq2etkkwas5zhx36.py
# Source Nodes: [shortcut_3, x_76, x_82, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_12
# x_76 => add_25, mul_37, mul_38, sub_11
# x_82 => add_27, mul_40, mul_41, sub_12
# x_86 => add_28
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhg53rjup2vakrkefjihglsnko7z3mhljxppwcxgtj7747htoxq.py
# Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_90 => add_30, mul_43, mul_44, sub_13
# x_94 => relu_13
# x_95 => convolution_20
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwadyas7xhchww52u7ukdtcgxdooblfckzoslpbmaxmplm3wp5p.py
# Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_12 => mean_3
# x_se_13 => convolution_21
# x_se_14 => relu_15
# x_se_15 => convolution_22
triton_poi_fused_convolution_mean_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 38
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/cai7bk4rpf3zmf6fxghcnfvq7xvkap3idiv6fi4kftcp5askgs52.py
# Source Nodes: [shortcut_4, x_103, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_4 => relu_16
# x_103 => add_34, mul_50, mul_51, sub_15
# x_108 => add_35
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chc54zk6xuufh5qpqvlta4l4wxpi6kprfnamdwze5y4nigzb4mhd.py
# Source Nodes: [x_157, x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_157 => add_51, mul_73, mul_74, sub_22
# x_161 => relu_25
# x_162 => convolution_35
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
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 368
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwg6ayvjwvsnbmidkvj76477cgpoqrskvc7gjpz4hv6ftx55rsoc.py
# Source Nodes: [x_163, x_167, x_se_24, x_se_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_163 => add_53, mul_76, mul_77, sub_23
# x_167 => relu_26
# x_se_24 => mean_6
# x_se_25 => convolution_36
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 368
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxm2lulifazjuclsffej6z5kq3c7w65mqi5if77rpca7j2xxgc3.py
# Source Nodes: [sigmoid_6, x_168, x_169, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_6 => sigmoid_6
# x_168 => mul_78
# x_169 => convolution_38
# x_se_24 => mean_6
# x_se_25 => convolution_36
# x_se_26 => relu_27
# x_se_27 => convolution_37
triton_poi_fused_convolution_mean_mul_relu_sigmoid_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4whln64sf3belqcsln7ll322ekfod6vhje37utmrsxomqxpbl7.py
# Source Nodes: [shortcut_7, x_170, x_176, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_7 => relu_28
# x_170 => add_55, mul_80, mul_81, sub_24
# x_176 => add_57, mul_83, mul_84, sub_25
# x_180 => add_58
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mjosnfkgijp2pngqyuoa7ocwpjlgh6ekvrez5pmng2ehy3fdi4.py
# Source Nodes: [x_184, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_184 => add_60, mul_86, mul_87, sub_26
# x_188 => relu_29
# x_189 => convolution_41
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mgksqaw233snn6hcnu72tk75i26xftzmwtw5zoym5wbrmgcfcl.py
# Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_28 => mean_7
# x_se_29 => convolution_42
# x_se_30 => relu_31
# x_se_31 => convolution_43
triton_poi_fused_convolution_mean_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldikbyx6e3ioai4j7qa3cs5qte7z6b3j5hkcdyfbtf3dh6ebfxe.py
# Source Nodes: [shortcut_8, x_197, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_8 => relu_32
# x_197 => add_64, mul_93, mul_94, sub_28
# x_202 => add_65
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
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
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsob2sbe3bqje4gxb5ldm6or2l5szr2v4hes5g7wg4theb7ttx2.py
# Source Nodes: [shortcut_9, x_219, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_9 => relu_36
# x_219 => add_71, mul_103, mul_104, sub_31
# x_224 => add_72
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c334hfspzzwrivdks5xrnoqkfblyuycvs5o45x47i7u5lodlh3fy.py
# Source Nodes: [x_307, x_312, x_315, x_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# x_307 => add_99, mul_143, mul_144, sub_43
# x_312 => add_100
# x_315 => relu_52
# x_318 => mean_13
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask & xmask, other=0.0)
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
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (24, ), (1, ))
    assert_size_stride(arg3_1, (24, ), (1, ))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (24, ), (1, ))
    assert_size_stride(arg6_1, (24, ), (1, ))
    assert_size_stride(arg7_1, (24, ), (1, ))
    assert_size_stride(arg8_1, (24, ), (1, ))
    assert_size_stride(arg9_1, (24, ), (1, ))
    assert_size_stride(arg10_1, (56, ), (1, ))
    assert_size_stride(arg11_1, (56, ), (1, ))
    assert_size_stride(arg12_1, (56, ), (1, ))
    assert_size_stride(arg13_1, (56, ), (1, ))
    assert_size_stride(arg14_1, (56, ), (1, ))
    assert_size_stride(arg15_1, (56, ), (1, ))
    assert_size_stride(arg16_1, (56, ), (1, ))
    assert_size_stride(arg17_1, (56, ), (1, ))
    assert_size_stride(arg18_1, (152, ), (1, ))
    assert_size_stride(arg19_1, (152, ), (1, ))
    assert_size_stride(arg20_1, (152, ), (1, ))
    assert_size_stride(arg21_1, (152, ), (1, ))
    assert_size_stride(arg22_1, (152, ), (1, ))
    assert_size_stride(arg23_1, (152, ), (1, ))
    assert_size_stride(arg24_1, (152, ), (1, ))
    assert_size_stride(arg25_1, (152, ), (1, ))
    assert_size_stride(arg26_1, (152, ), (1, ))
    assert_size_stride(arg27_1, (152, ), (1, ))
    assert_size_stride(arg28_1, (152, ), (1, ))
    assert_size_stride(arg29_1, (152, ), (1, ))
    assert_size_stride(arg30_1, (152, ), (1, ))
    assert_size_stride(arg31_1, (152, ), (1, ))
    assert_size_stride(arg32_1, (152, ), (1, ))
    assert_size_stride(arg33_1, (152, ), (1, ))
    assert_size_stride(arg34_1, (152, ), (1, ))
    assert_size_stride(arg35_1, (152, ), (1, ))
    assert_size_stride(arg36_1, (152, ), (1, ))
    assert_size_stride(arg37_1, (152, ), (1, ))
    assert_size_stride(arg38_1, (152, ), (1, ))
    assert_size_stride(arg39_1, (152, ), (1, ))
    assert_size_stride(arg40_1, (152, ), (1, ))
    assert_size_stride(arg41_1, (152, ), (1, ))
    assert_size_stride(arg42_1, (152, ), (1, ))
    assert_size_stride(arg43_1, (152, ), (1, ))
    assert_size_stride(arg44_1, (368, ), (1, ))
    assert_size_stride(arg45_1, (368, ), (1, ))
    assert_size_stride(arg46_1, (368, ), (1, ))
    assert_size_stride(arg47_1, (368, ), (1, ))
    assert_size_stride(arg48_1, (368, ), (1, ))
    assert_size_stride(arg49_1, (368, ), (1, ))
    assert_size_stride(arg50_1, (368, ), (1, ))
    assert_size_stride(arg51_1, (368, ), (1, ))
    assert_size_stride(arg52_1, (368, ), (1, ))
    assert_size_stride(arg53_1, (368, ), (1, ))
    assert_size_stride(arg54_1, (368, ), (1, ))
    assert_size_stride(arg55_1, (368, ), (1, ))
    assert_size_stride(arg56_1, (368, ), (1, ))
    assert_size_stride(arg57_1, (368, ), (1, ))
    assert_size_stride(arg58_1, (368, ), (1, ))
    assert_size_stride(arg59_1, (368, ), (1, ))
    assert_size_stride(arg60_1, (368, ), (1, ))
    assert_size_stride(arg61_1, (368, ), (1, ))
    assert_size_stride(arg62_1, (368, ), (1, ))
    assert_size_stride(arg63_1, (368, ), (1, ))
    assert_size_stride(arg64_1, (368, ), (1, ))
    assert_size_stride(arg65_1, (368, ), (1, ))
    assert_size_stride(arg66_1, (368, ), (1, ))
    assert_size_stride(arg67_1, (368, ), (1, ))
    assert_size_stride(arg68_1, (368, ), (1, ))
    assert_size_stride(arg69_1, (368, ), (1, ))
    assert_size_stride(arg70_1, (368, ), (1, ))
    assert_size_stride(arg71_1, (368, ), (1, ))
    assert_size_stride(arg72_1, (368, ), (1, ))
    assert_size_stride(arg73_1, (368, ), (1, ))
    assert_size_stride(arg74_1, (368, ), (1, ))
    assert_size_stride(arg75_1, (368, ), (1, ))
    assert_size_stride(arg76_1, (368, ), (1, ))
    assert_size_stride(arg77_1, (368, ), (1, ))
    assert_size_stride(arg78_1, (368, ), (1, ))
    assert_size_stride(arg79_1, (368, ), (1, ))
    assert_size_stride(arg80_1, (368, ), (1, ))
    assert_size_stride(arg81_1, (368, ), (1, ))
    assert_size_stride(arg82_1, (368, ), (1, ))
    assert_size_stride(arg83_1, (368, ), (1, ))
    assert_size_stride(arg84_1, (368, ), (1, ))
    assert_size_stride(arg85_1, (368, ), (1, ))
    assert_size_stride(arg86_1, (368, ), (1, ))
    assert_size_stride(arg87_1, (368, ), (1, ))
    assert_size_stride(arg88_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg89_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg90_1, (24, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg91_1, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg92_1, (8, ), (1, ))
    assert_size_stride(arg93_1, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg94_1, (24, ), (1, ))
    assert_size_stride(arg95_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg96_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg97_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg98_1, (56, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg99_1, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg100_1, (6, ), (1, ))
    assert_size_stride(arg101_1, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg102_1, (56, ), (1, ))
    assert_size_stride(arg103_1, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg104_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg105_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg106_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg107_1, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg108_1, (14, ), (1, ))
    assert_size_stride(arg109_1, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg110_1, (152, ), (1, ))
    assert_size_stride(arg111_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg112_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg113_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg114_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg115_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg116_1, (38, ), (1, ))
    assert_size_stride(arg117_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg118_1, (152, ), (1, ))
    assert_size_stride(arg119_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg120_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg121_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg122_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg123_1, (38, ), (1, ))
    assert_size_stride(arg124_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg125_1, (152, ), (1, ))
    assert_size_stride(arg126_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg127_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg128_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg129_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg130_1, (38, ), (1, ))
    assert_size_stride(arg131_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg132_1, (152, ), (1, ))
    assert_size_stride(arg133_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg134_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg135_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg136_1, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg137_1, (38, ), (1, ))
    assert_size_stride(arg138_1, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg139_1, (368, ), (1, ))
    assert_size_stride(arg140_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg141_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg142_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg143_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg144_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg145_1, (92, ), (1, ))
    assert_size_stride(arg146_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg147_1, (368, ), (1, ))
    assert_size_stride(arg148_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg149_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg150_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg151_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg152_1, (92, ), (1, ))
    assert_size_stride(arg153_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg154_1, (368, ), (1, ))
    assert_size_stride(arg155_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg156_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg157_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg158_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg159_1, (92, ), (1, ))
    assert_size_stride(arg160_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg161_1, (368, ), (1, ))
    assert_size_stride(arg162_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg163_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg164_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg165_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg166_1, (92, ), (1, ))
    assert_size_stride(arg167_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg168_1, (368, ), (1, ))
    assert_size_stride(arg169_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg170_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg171_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg172_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg173_1, (92, ), (1, ))
    assert_size_stride(arg174_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg175_1, (368, ), (1, ))
    assert_size_stride(arg176_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg177_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg178_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg179_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg180_1, (92, ), (1, ))
    assert_size_stride(arg181_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg182_1, (368, ), (1, ))
    assert_size_stride(arg183_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg184_1, (1000, 368), (368, 1))
    assert_size_stride(arg185_1, (1000, ), (1, ))
    assert_size_stride(arg186_1, (32, ), (1, ))
    assert_size_stride(arg187_1, (32, ), (1, ))
    assert_size_stride(arg188_1, (24, ), (1, ))
    assert_size_stride(arg189_1, (24, ), (1, ))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (24, ), (1, ))
    assert_size_stride(arg194_1, (24, ), (1, ))
    assert_size_stride(arg195_1, (24, ), (1, ))
    assert_size_stride(arg196_1, (56, ), (1, ))
    assert_size_stride(arg197_1, (56, ), (1, ))
    assert_size_stride(arg198_1, (56, ), (1, ))
    assert_size_stride(arg199_1, (56, ), (1, ))
    assert_size_stride(arg200_1, (56, ), (1, ))
    assert_size_stride(arg201_1, (56, ), (1, ))
    assert_size_stride(arg202_1, (56, ), (1, ))
    assert_size_stride(arg203_1, (56, ), (1, ))
    assert_size_stride(arg204_1, (152, ), (1, ))
    assert_size_stride(arg205_1, (152, ), (1, ))
    assert_size_stride(arg206_1, (152, ), (1, ))
    assert_size_stride(arg207_1, (152, ), (1, ))
    assert_size_stride(arg208_1, (152, ), (1, ))
    assert_size_stride(arg209_1, (152, ), (1, ))
    assert_size_stride(arg210_1, (152, ), (1, ))
    assert_size_stride(arg211_1, (152, ), (1, ))
    assert_size_stride(arg212_1, (152, ), (1, ))
    assert_size_stride(arg213_1, (152, ), (1, ))
    assert_size_stride(arg214_1, (152, ), (1, ))
    assert_size_stride(arg215_1, (152, ), (1, ))
    assert_size_stride(arg216_1, (152, ), (1, ))
    assert_size_stride(arg217_1, (152, ), (1, ))
    assert_size_stride(arg218_1, (152, ), (1, ))
    assert_size_stride(arg219_1, (152, ), (1, ))
    assert_size_stride(arg220_1, (152, ), (1, ))
    assert_size_stride(arg221_1, (152, ), (1, ))
    assert_size_stride(arg222_1, (152, ), (1, ))
    assert_size_stride(arg223_1, (152, ), (1, ))
    assert_size_stride(arg224_1, (152, ), (1, ))
    assert_size_stride(arg225_1, (152, ), (1, ))
    assert_size_stride(arg226_1, (152, ), (1, ))
    assert_size_stride(arg227_1, (152, ), (1, ))
    assert_size_stride(arg228_1, (152, ), (1, ))
    assert_size_stride(arg229_1, (152, ), (1, ))
    assert_size_stride(arg230_1, (368, ), (1, ))
    assert_size_stride(arg231_1, (368, ), (1, ))
    assert_size_stride(arg232_1, (368, ), (1, ))
    assert_size_stride(arg233_1, (368, ), (1, ))
    assert_size_stride(arg234_1, (368, ), (1, ))
    assert_size_stride(arg235_1, (368, ), (1, ))
    assert_size_stride(arg236_1, (368, ), (1, ))
    assert_size_stride(arg237_1, (368, ), (1, ))
    assert_size_stride(arg238_1, (368, ), (1, ))
    assert_size_stride(arg239_1, (368, ), (1, ))
    assert_size_stride(arg240_1, (368, ), (1, ))
    assert_size_stride(arg241_1, (368, ), (1, ))
    assert_size_stride(arg242_1, (368, ), (1, ))
    assert_size_stride(arg243_1, (368, ), (1, ))
    assert_size_stride(arg244_1, (368, ), (1, ))
    assert_size_stride(arg245_1, (368, ), (1, ))
    assert_size_stride(arg246_1, (368, ), (1, ))
    assert_size_stride(arg247_1, (368, ), (1, ))
    assert_size_stride(arg248_1, (368, ), (1, ))
    assert_size_stride(arg249_1, (368, ), (1, ))
    assert_size_stride(arg250_1, (368, ), (1, ))
    assert_size_stride(arg251_1, (368, ), (1, ))
    assert_size_stride(arg252_1, (368, ), (1, ))
    assert_size_stride(arg253_1, (368, ), (1, ))
    assert_size_stride(arg254_1, (368, ), (1, ))
    assert_size_stride(arg255_1, (368, ), (1, ))
    assert_size_stride(arg256_1, (368, ), (1, ))
    assert_size_stride(arg257_1, (368, ), (1, ))
    assert_size_stride(arg258_1, (368, ), (1, ))
    assert_size_stride(arg259_1, (368, ), (1, ))
    assert_size_stride(arg260_1, (368, ), (1, ))
    assert_size_stride(arg261_1, (368, ), (1, ))
    assert_size_stride(arg262_1, (368, ), (1, ))
    assert_size_stride(arg263_1, (368, ), (1, ))
    assert_size_stride(arg264_1, (368, ), (1, ))
    assert_size_stride(arg265_1, (368, ), (1, ))
    assert_size_stride(arg266_1, (368, ), (1, ))
    assert_size_stride(arg267_1, (368, ), (1, ))
    assert_size_stride(arg268_1, (368, ), (1, ))
    assert_size_stride(arg269_1, (368, ), (1, ))
    assert_size_stride(arg270_1, (368, ), (1, ))
    assert_size_stride(arg271_1, (368, ), (1, ))
    assert_size_stride(arg272_1, (368, ), (1, ))
    assert_size_stride(arg273_1, (368, ), (1, ))
    assert_size_stride(arg274_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg274_1, arg88_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg274_1
        del arg88_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg186_1, arg187_1, arg0_1, arg1_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg0_1
        del arg186_1
        del arg187_1
        del arg1_1
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 24, 112, 112), (301056, 12544, 112, 1))
        del arg89_1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf3, arg188_1, arg189_1, arg2_1, arg3_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg188_1
        del arg189_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg90_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf4, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg90_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided((8, 24, 1, 1), (24, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf6, (8, 24, 1, 1), (24, 1, 1, 1), 0); del buf6  # reuse
        # Source Nodes: [x_13, x_17, x_se, x_se_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2.run(buf5, buf7, arg190_1, arg191_1, arg4_1, arg5_1, 192, 3136, grid=grid(192), stream=stream0)
        del arg190_1
        del arg191_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf8 = extern_kernels.convolution(buf7, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg91_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_3.run(buf9, arg92_1, 64, grid=grid(64), stream=stream0)
        del arg92_1
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg93_1
        del buf9
        buf11 = buf5; del buf5  # reuse
        # Source Nodes: [sigmoid, x_18, x_19, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_4.run(buf11, buf10, arg94_1, 602112, grid=grid(602112), stream=stream0)
        del arg94_1
        del buf10
        # Source Nodes: [sigmoid, x_18, x_19, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf12 = extern_kernels.convolution(buf11, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg95_1
        del buf11
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf1, arg96_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg96_1
        del buf1
        buf14 = buf12; del buf12  # reuse
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf15, arg192_1, arg193_1, arg6_1, arg7_1, buf13, arg194_1, arg195_1, arg8_1, arg9_1, 602112, grid=grid(602112), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf13
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 56, 56, 56), (175616, 3136, 56, 1))
        del arg97_1
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [x_35, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf17, arg196_1, arg197_1, arg10_1, arg11_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg10_1
        del arg11_1
        del arg196_1
        del arg197_1
        # Source Nodes: [x_35, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg98_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=7, bias=None)
        assert_size_stride(buf18, (8, 56, 28, 28), (43904, 784, 28, 1))
        del arg98_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided((8, 56, 1, 1), (56, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf20, (8, 56, 1, 1), (56, 1, 1, 1), 0); del buf20  # reuse
        # Source Nodes: [x_41, x_45, x_se_4, x_se_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7.run(buf19, buf21, arg198_1, arg199_1, arg12_1, arg13_1, 448, 784, grid=grid(448), stream=stream0)
        del arg12_1
        del arg13_1
        del arg198_1
        del arg199_1
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf22 = extern_kernels.convolution(buf21, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg99_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_8.run(buf23, arg100_1, 48, grid=grid(48), stream=stream0)
        del arg100_1
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 56, 1, 1), (56, 1, 1, 1))
        del arg101_1
        del buf23
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [sigmoid_1, x_46, x_47, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_9.run(buf25, buf24, arg102_1, 351232, grid=grid(351232), stream=stream0)
        del arg102_1
        del buf24
        # Source Nodes: [sigmoid_1, x_46, x_47, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf26 = extern_kernels.convolution(buf25, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 56, 28, 28), (43904, 784, 28, 1))
        del arg103_1
        del buf25
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf15, arg104_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 56, 28, 28), (43904, 784, 28, 1))
        del arg104_1
        del buf15
        buf28 = buf26; del buf26  # reuse
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [shortcut_2, x_48, x_54, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf29, arg200_1, arg201_1, arg14_1, arg15_1, buf27, arg202_1, arg203_1, arg16_1, arg17_1, 351232, grid=grid(351232), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del buf27
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 152, 28, 28), (119168, 784, 28, 1))
        del arg105_1
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [x_63, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf31, arg204_1, arg205_1, arg18_1, arg19_1, 953344, grid=grid(953344), stream=stream0)
        del arg18_1
        del arg19_1
        del arg204_1
        del arg205_1
        # Source Nodes: [x_63, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf32 = extern_kernels.convolution(buf31, arg106_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf32, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg106_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf35 = reinterpret_tensor(buf34, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf34  # reuse
        # Source Nodes: [x_69, x_73, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12.run(buf33, buf35, arg206_1, arg207_1, arg20_1, arg21_1, 1216, 196, grid=grid(1216), stream=stream0)
        del arg206_1
        del arg207_1
        del arg20_1
        del arg21_1
        # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
        buf36 = extern_kernels.convolution(buf35, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 14, 1, 1), (14, 1, 1, 1))
        del arg107_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_13.run(buf37, arg108_1, 112, grid=grid(112), stream=stream0)
        del arg108_1
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf38 = extern_kernels.convolution(buf37, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg109_1
        del buf37
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [sigmoid_2, x_74, x_75, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_14.run(buf39, buf38, arg110_1, 238336, grid=grid(238336), stream=stream0)
        del arg110_1
        # Source Nodes: [sigmoid_2, x_74, x_75, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf40 = extern_kernels.convolution(buf39, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg111_1
        del buf39
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf29, arg112_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg112_1
        del buf29
        buf42 = buf40; del buf40  # reuse
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [shortcut_3, x_76, x_82, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf43, arg208_1, arg209_1, arg22_1, arg23_1, buf41, arg210_1, arg211_1, arg24_1, arg25_1, 238336, grid=grid(238336), stream=stream0)
        del arg208_1
        del arg209_1
        del arg210_1
        del arg211_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf41
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg113_1
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf45, arg212_1, arg213_1, arg26_1, arg27_1, 238336, grid=grid(238336), stream=stream0)
        del arg212_1
        del arg213_1
        del arg26_1
        del arg27_1
        # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf46, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg114_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        buf48 = reinterpret_tensor(buf38, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf38  # reuse
        buf49 = reinterpret_tensor(buf48, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf48  # reuse
        # Source Nodes: [x_100, x_96, x_se_12, x_se_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12.run(buf47, buf49, arg214_1, arg215_1, arg28_1, arg29_1, 1216, 196, grid=grid(1216), stream=stream0)
        del arg214_1
        del arg215_1
        del arg28_1
        del arg29_1
        # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
        buf50 = extern_kernels.convolution(buf49, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg115_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_17.run(buf51, arg116_1, 304, grid=grid(304), stream=stream0)
        del arg116_1
        # Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg117_1
        del buf51
        buf53 = buf47; del buf47  # reuse
        # Source Nodes: [sigmoid_3, x_101, x_102, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_14.run(buf53, buf52, arg118_1, 238336, grid=grid(238336), stream=stream0)
        del arg118_1
        # Source Nodes: [sigmoid_3, x_101, x_102, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf54 = extern_kernels.convolution(buf53, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg119_1
        del buf53
        buf55 = buf43; del buf43  # reuse
        # Source Nodes: [shortcut_4, x_103, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf55, buf54, arg216_1, arg217_1, arg30_1, arg31_1, 238336, grid=grid(238336), stream=stream0)
        del arg216_1
        del arg217_1
        del arg30_1
        del arg31_1
        del buf54
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg120_1
        buf57 = buf56; del buf56  # reuse
        # Source Nodes: [x_112, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf57, arg218_1, arg219_1, arg32_1, arg33_1, 238336, grid=grid(238336), stream=stream0)
        del arg218_1
        del arg219_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_112, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg121_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf58, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg121_1
        del buf57
        buf59 = buf58; del buf58  # reuse
        buf60 = reinterpret_tensor(buf52, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf52  # reuse
        buf61 = reinterpret_tensor(buf60, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf60  # reuse
        # Source Nodes: [x_118, x_122, x_se_16, x_se_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12.run(buf59, buf61, arg220_1, arg221_1, arg34_1, arg35_1, 1216, 196, grid=grid(1216), stream=stream0)
        del arg220_1
        del arg221_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
        buf62 = extern_kernels.convolution(buf61, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg122_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Source Nodes: [x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_17.run(buf63, arg123_1, 304, grid=grid(304), stream=stream0)
        del arg123_1
        # Source Nodes: [x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg124_1
        del buf63
        buf65 = buf59; del buf59  # reuse
        # Source Nodes: [sigmoid_4, x_123, x_124, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_14.run(buf65, buf64, arg125_1, 238336, grid=grid(238336), stream=stream0)
        del arg125_1
        # Source Nodes: [sigmoid_4, x_123, x_124, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf66 = extern_kernels.convolution(buf65, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg126_1
        del buf65
        buf67 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_5, x_125, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf67, buf66, arg222_1, arg223_1, arg36_1, arg37_1, 238336, grid=grid(238336), stream=stream0)
        del arg222_1
        del arg223_1
        del arg36_1
        del arg37_1
        del buf66
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg127_1
        buf69 = buf68; del buf68  # reuse
        # Source Nodes: [x_134, x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf69, arg224_1, arg225_1, arg38_1, arg39_1, 238336, grid=grid(238336), stream=stream0)
        del arg224_1
        del arg225_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_134, x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf69, arg128_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf70, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg128_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        buf72 = reinterpret_tensor(buf64, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf64  # reuse
        buf73 = reinterpret_tensor(buf72, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf72  # reuse
        # Source Nodes: [x_140, x_144, x_se_20, x_se_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12.run(buf71, buf73, arg226_1, arg227_1, arg40_1, arg41_1, 1216, 196, grid=grid(1216), stream=stream0)
        del arg226_1
        del arg227_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
        buf74 = extern_kernels.convolution(buf73, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg129_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_17.run(buf75, arg130_1, 304, grid=grid(304), stream=stream0)
        del arg130_1
        # Source Nodes: [x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg131_1
        del buf75
        buf77 = buf71; del buf71  # reuse
        # Source Nodes: [sigmoid_5, x_145, x_146, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_14.run(buf77, buf76, arg132_1, 238336, grid=grid(238336), stream=stream0)
        del arg132_1
        del buf76
        # Source Nodes: [sigmoid_5, x_145, x_146, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf78 = extern_kernels.convolution(buf77, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg133_1
        del buf77
        buf79 = buf67; del buf67  # reuse
        # Source Nodes: [shortcut_6, x_147, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf79, buf78, arg228_1, arg229_1, arg42_1, arg43_1, 238336, grid=grid(238336), stream=stream0)
        del arg228_1
        del arg229_1
        del arg42_1
        del arg43_1
        del buf78
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 368, 14, 14), (72128, 196, 14, 1))
        del arg134_1
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_157, x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf81, arg230_1, arg231_1, arg44_1, arg45_1, 577024, grid=grid(577024), stream=stream0)
        del arg230_1
        del arg231_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_157, x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg135_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf82, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg135_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        buf84 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf85 = reinterpret_tensor(buf84, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf84  # reuse
        # Source Nodes: [x_163, x_167, x_se_24, x_se_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf83, buf85, arg232_1, arg233_1, arg46_1, arg47_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg232_1
        del arg233_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
        buf86 = extern_kernels.convolution(buf85, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg136_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Source Nodes: [x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_17.run(buf87, arg137_1, 304, grid=grid(304), stream=stream0)
        del arg137_1
        # Source Nodes: [x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg138_1
        del buf87
        buf89 = buf83; del buf83  # reuse
        # Source Nodes: [sigmoid_6, x_168, x_169, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf89, buf88, arg139_1, 144256, grid=grid(144256), stream=stream0)
        del arg139_1
        # Source Nodes: [sigmoid_6, x_168, x_169, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf90 = extern_kernels.convolution(buf89, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg140_1
        del buf89
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf79, arg141_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg141_1
        del buf79
        buf92 = buf90; del buf90  # reuse
        buf93 = buf92; del buf92  # reuse
        # Source Nodes: [shortcut_7, x_170, x_176, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf93, arg234_1, arg235_1, arg48_1, arg49_1, buf91, arg236_1, arg237_1, arg50_1, arg51_1, 144256, grid=grid(144256), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del arg237_1
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        del buf91
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg142_1
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [x_184, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf95, arg238_1, arg239_1, arg52_1, arg53_1, 144256, grid=grid(144256), stream=stream0)
        del arg238_1
        del arg239_1
        del arg52_1
        del arg53_1
        # Source Nodes: [x_184, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf96 = extern_kernels.convolution(buf95, arg143_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf96, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg143_1
        del buf95
        buf97 = buf96; del buf96  # reuse
        buf98 = reinterpret_tensor(buf88, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf88  # reuse
        buf99 = reinterpret_tensor(buf98, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf98  # reuse
        # Source Nodes: [x_190, x_194, x_se_28, x_se_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf97, buf99, arg240_1, arg241_1, arg54_1, arg55_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg240_1
        del arg241_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
        buf100 = extern_kernels.convolution(buf99, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg144_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf101, arg145_1, 736, grid=grid(736), stream=stream0)
        del arg145_1
        # Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf102 = extern_kernels.convolution(buf101, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg146_1
        del buf101
        buf103 = buf97; del buf97  # reuse
        # Source Nodes: [sigmoid_7, x_195, x_196, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf103, buf102, arg147_1, 144256, grid=grid(144256), stream=stream0)
        del arg147_1
        # Source Nodes: [sigmoid_7, x_195, x_196, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf104 = extern_kernels.convolution(buf103, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg148_1
        del buf103
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [shortcut_8, x_197, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf105, arg242_1, arg243_1, arg56_1, arg57_1, buf93, 144256, grid=grid(144256), stream=stream0)
        del arg242_1
        del arg243_1
        del arg56_1
        del arg57_1
        del buf93
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg149_1
        buf107 = buf106; del buf106  # reuse
        # Source Nodes: [x_206, x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf107, arg244_1, arg245_1, arg58_1, arg59_1, 144256, grid=grid(144256), stream=stream0)
        del arg244_1
        del arg245_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_206, x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf108 = extern_kernels.convolution(buf107, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf108, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg150_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        buf110 = reinterpret_tensor(buf102, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf102  # reuse
        buf111 = reinterpret_tensor(buf110, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf110  # reuse
        # Source Nodes: [x_212, x_216, x_se_32, x_se_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf109, buf111, arg246_1, arg247_1, arg60_1, arg61_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg246_1
        del arg247_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
        buf112 = extern_kernels.convolution(buf111, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg151_1
        del buf111
        buf113 = buf112; del buf112  # reuse
        # Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf113, arg152_1, 736, grid=grid(736), stream=stream0)
        del arg152_1
        # Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf114 = extern_kernels.convolution(buf113, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg153_1
        del buf113
        buf115 = buf109; del buf109  # reuse
        # Source Nodes: [sigmoid_8, x_217, x_218, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf115, buf114, arg154_1, 144256, grid=grid(144256), stream=stream0)
        del arg154_1
        # Source Nodes: [sigmoid_8, x_217, x_218, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf116 = extern_kernels.convolution(buf115, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg155_1
        del buf115
        buf117 = buf105; del buf105  # reuse
        # Source Nodes: [shortcut_9, x_219, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf117, buf116, arg248_1, arg249_1, arg62_1, arg63_1, 144256, grid=grid(144256), stream=stream0)
        del arg248_1
        del arg249_1
        del arg62_1
        del arg63_1
        del buf116
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg156_1
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [x_228, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf119, arg250_1, arg251_1, arg64_1, arg65_1, 144256, grid=grid(144256), stream=stream0)
        del arg250_1
        del arg251_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_228, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf120 = extern_kernels.convolution(buf119, arg157_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf120, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg157_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        buf122 = reinterpret_tensor(buf114, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf114  # reuse
        buf123 = reinterpret_tensor(buf122, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf122  # reuse
        # Source Nodes: [x_234, x_238, x_se_36, x_se_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf121, buf123, arg252_1, arg253_1, arg66_1, arg67_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg252_1
        del arg253_1
        del arg66_1
        del arg67_1
        # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
        buf124 = extern_kernels.convolution(buf123, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg158_1
        del buf123
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf125, arg159_1, 736, grid=grid(736), stream=stream0)
        del arg159_1
        # Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg160_1
        del buf125
        buf127 = buf121; del buf121  # reuse
        # Source Nodes: [sigmoid_9, x_239, x_240, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf127, buf126, arg161_1, 144256, grid=grid(144256), stream=stream0)
        del arg161_1
        # Source Nodes: [sigmoid_9, x_239, x_240, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf128 = extern_kernels.convolution(buf127, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg162_1
        del buf127
        buf129 = buf117; del buf117  # reuse
        # Source Nodes: [shortcut_10, x_241, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf129, buf128, arg254_1, arg255_1, arg68_1, arg69_1, 144256, grid=grid(144256), stream=stream0)
        del arg254_1
        del arg255_1
        del arg68_1
        del arg69_1
        del buf128
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg163_1
        buf131 = buf130; del buf130  # reuse
        # Source Nodes: [x_250, x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf131, arg256_1, arg257_1, arg70_1, arg71_1, 144256, grid=grid(144256), stream=stream0)
        del arg256_1
        del arg257_1
        del arg70_1
        del arg71_1
        # Source Nodes: [x_250, x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf132 = extern_kernels.convolution(buf131, arg164_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf132, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg164_1
        del buf131
        buf133 = buf132; del buf132  # reuse
        buf134 = reinterpret_tensor(buf126, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf126  # reuse
        buf135 = reinterpret_tensor(buf134, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf134  # reuse
        # Source Nodes: [x_256, x_260, x_se_40, x_se_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf133, buf135, arg258_1, arg259_1, arg72_1, arg73_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg258_1
        del arg259_1
        del arg72_1
        del arg73_1
        # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
        buf136 = extern_kernels.convolution(buf135, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg165_1
        del buf135
        buf137 = buf136; del buf136  # reuse
        # Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf137, arg166_1, 736, grid=grid(736), stream=stream0)
        del arg166_1
        # Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf138 = extern_kernels.convolution(buf137, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg167_1
        del buf137
        buf139 = buf133; del buf133  # reuse
        # Source Nodes: [sigmoid_10, x_261, x_262, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf139, buf138, arg168_1, 144256, grid=grid(144256), stream=stream0)
        del arg168_1
        # Source Nodes: [sigmoid_10, x_261, x_262, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf140 = extern_kernels.convolution(buf139, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg169_1
        del buf139
        buf141 = buf129; del buf129  # reuse
        # Source Nodes: [shortcut_11, x_263, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf141, buf140, arg260_1, arg261_1, arg74_1, arg75_1, 144256, grid=grid(144256), stream=stream0)
        del arg260_1
        del arg261_1
        del arg74_1
        del arg75_1
        del buf140
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg170_1
        buf143 = buf142; del buf142  # reuse
        # Source Nodes: [x_272, x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf143, arg262_1, arg263_1, arg76_1, arg77_1, 144256, grid=grid(144256), stream=stream0)
        del arg262_1
        del arg263_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_272, x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf144 = extern_kernels.convolution(buf143, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf144, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg171_1
        del buf143
        buf145 = buf144; del buf144  # reuse
        buf146 = reinterpret_tensor(buf138, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf138  # reuse
        buf147 = reinterpret_tensor(buf146, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf146  # reuse
        # Source Nodes: [x_278, x_282, x_se_44, x_se_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf145, buf147, arg264_1, arg265_1, arg78_1, arg79_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg264_1
        del arg265_1
        del arg78_1
        del arg79_1
        # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
        buf148 = extern_kernels.convolution(buf147, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg172_1
        del buf147
        buf149 = buf148; del buf148  # reuse
        # Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf149, arg173_1, 736, grid=grid(736), stream=stream0)
        del arg173_1
        # Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf150 = extern_kernels.convolution(buf149, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg174_1
        del buf149
        buf151 = buf145; del buf145  # reuse
        # Source Nodes: [sigmoid_11, x_283, x_284, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf151, buf150, arg175_1, 144256, grid=grid(144256), stream=stream0)
        del arg175_1
        # Source Nodes: [sigmoid_11, x_283, x_284, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf152 = extern_kernels.convolution(buf151, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg176_1
        del buf151
        buf153 = buf141; del buf141  # reuse
        # Source Nodes: [shortcut_12, x_285, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf153, buf152, arg266_1, arg267_1, arg80_1, arg81_1, 144256, grid=grid(144256), stream=stream0)
        del arg266_1
        del arg267_1
        del arg80_1
        del arg81_1
        del buf152
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg177_1
        buf155 = buf154; del buf154  # reuse
        # Source Nodes: [x_294, x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf155, arg268_1, arg269_1, arg82_1, arg83_1, 144256, grid=grid(144256), stream=stream0)
        del arg268_1
        del arg269_1
        del arg82_1
        del arg83_1
        # Source Nodes: [x_294, x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf156 = extern_kernels.convolution(buf155, arg178_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf156, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg178_1
        del buf155
        buf157 = buf156; del buf156  # reuse
        buf158 = reinterpret_tensor(buf150, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf150  # reuse
        buf159 = reinterpret_tensor(buf158, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf158  # reuse
        # Source Nodes: [x_300, x_304, x_se_48, x_se_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_20.run(buf157, buf159, arg270_1, arg271_1, arg84_1, arg85_1, 2944, 49, grid=grid(2944), stream=stream0)
        del arg270_1
        del arg271_1
        del arg84_1
        del arg85_1
        # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
        buf160 = extern_kernels.convolution(buf159, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg179_1
        del buf159
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf161, arg180_1, 736, grid=grid(736), stream=stream0)
        del arg180_1
        # Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf162 = extern_kernels.convolution(buf161, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg181_1
        del buf161
        buf163 = buf157; del buf157  # reuse
        # Source Nodes: [sigmoid_12, x_305, x_306, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_21.run(buf163, buf162, arg182_1, 144256, grid=grid(144256), stream=stream0)
        del arg182_1
        # Source Nodes: [sigmoid_12, x_305, x_306, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf164 = extern_kernels.convolution(buf163, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 368, 7, 7), (18032, 49, 7, 1))
        del arg183_1
        del buf163
        buf165 = reinterpret_tensor(buf162, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf162  # reuse
        buf166 = reinterpret_tensor(buf165, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf165  # reuse
        # Source Nodes: [x_307, x_312, x_315, x_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_27.run(buf166, buf164, arg272_1, arg273_1, arg86_1, arg87_1, buf153, 2944, 49, grid=grid(2944), stream=stream0)
        del arg272_1
        del arg273_1
        del arg86_1
        del arg87_1
        del buf153
        del buf164
        buf167 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg185_1, reinterpret_tensor(buf166, (8, 368), (368, 1), 0), reinterpret_tensor(arg184_1, (368, 1000), (1, 368), 0), alpha=1, beta=1, out=buf167)
        del arg184_1
        del arg185_1
        return (buf167, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((24, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((56, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1000, 368), (368, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
