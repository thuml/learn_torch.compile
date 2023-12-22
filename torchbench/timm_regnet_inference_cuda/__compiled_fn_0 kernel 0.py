
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


# kernel path: /tmp/torchinductor_youkaichao/f3/cf345hieddpscxutlhetdhrpqh4xmsymlxgxcr2q2j35oyerkvig.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg4zadrbqvuv6bqjgssnbtlkbpoykagdsaqhft6o3cle37hiriw.py
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
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11239424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcfsmsbc6lnbno5w6h42pfsnqfgdw56ub23zqj4jjpqqujdojim.py
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
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_youkaichao/iv/civm7v7jozpzhqf254dvdtdqkqwm6ovv26rw4tqlukkuovkabviq.py
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
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/na/cnagcccy5pr4rwojie6x5jqjlbvupvhnen5mhncyo55rffbkfgbv.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 224
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca62shhlwn2yugk3z2num3qwpxnkvlf5sm3yk5nbxzsca5qv7esi.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/i7/ci757k63526u76ou33qomyl4axk3g76phxy7atflywkgyebsmfi7.py
# Source Nodes: [x_34, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_34 => add_12, mul_17, mul_18, sub_5
# x_38 => relu_5
# x_39 => convolution_8
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tvgrw6kh2ea63b5mgxr4bd5eixo2pkuhw5pm5qxj4lbswpewvh.py
# Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_4 => mean_1
# x_se_5 => convolution_9
# x_se_6 => relu_7
# x_se_7 => convolution_10
triton_poi_fused_convolution_mean_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgmddblicmjszr7bly6rw4okb54f5vl4szp6ubl3wvr3tkkzmlq.py
# Source Nodes: [shortcut_2, x_47, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_2 => relu_8
# x_47 => add_16, mul_24, mul_25, sub_7
# x_52 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czoubmbiscnbamn6es3h4vs7yebdoxlfzgotbj3quh46wyd5vjs7.py
# Source Nodes: [x_57, x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_57 => add_19, mul_27, mul_28, sub_8
# x_61 => relu_9
# x_62 => convolution_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkcnbze75moxsrldg6sdcthyn676tue4o4ncjvwj5mbkdtc4b54.py
# Source Nodes: [x_63, x_67, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_63 => add_21, mul_30, mul_31, sub_9
# x_67 => relu_10
# x_se_8 => mean_2
# x_se_9 => convolution_14
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1792
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
    x0 = xindex % 448
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


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgrhct6gvfku3uslkztf5byp6sagxsbc3mj5oytnwjajvpxkulc.py
# Source Nodes: [sigmoid_2, x_68, x_69, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_2 => sigmoid_2
# x_68 => mul_32
# x_69 => convolution_16
# x_se_10 => relu_11
# x_se_11 => convolution_15
# x_se_8 => mean_2
# x_se_9 => convolution_14
triton_poi_fused_convolution_mean_mul_relu_sigmoid_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwhq5dfo2x2lxgpuxk2gunayz75smw2icxjetiotjl6sk7mcvmx.py
# Source Nodes: [shortcut_3, x_70, x_76, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_12
# x_70 => add_23, mul_34, mul_35, sub_10
# x_76 => add_25, mul_37, mul_38, sub_11
# x_80 => add_26
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2d76fcusdrpv6tyi7vgbkcfhcbp4uq45ejveqwfbvu3l6k4akg3.py
# Source Nodes: [x_84, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_84 => add_28, mul_40, mul_41, sub_12
# x_88 => relu_13
# x_89 => convolution_19
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/ae/caex7dpbz7ymsqnlfatgoe44ghuphuywf2fexhmsyubo7c25q2mw.py
# Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_12 => mean_3
# x_se_13 => convolution_20
# x_se_14 => relu_15
# x_se_15 => convolution_21
triton_poi_fused_convolution_mean_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cwko3tlhd3qzima3hiktcamysa3rw7ff5lx5mn6edtuc6byojb.py
# Source Nodes: [shortcut_4, x_102, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_4 => relu_16
# x_102 => add_33
# x_97 => add_32, mul_47, mul_48, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5plwqbth233jqu6vxpgggpbdmgdsdwke2fjgnpdewwg6yvx3jt.py
# Source Nodes: [x_173, x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_173 => add_56, mul_80, mul_81, sub_24
# x_177 => relu_29
# x_178 => convolution_39
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2jbf57jfvgiqywj3uls7rl7dsm7qcrvustohwl4rrn2i74iyjy.py
# Source Nodes: [x_179, x_183, x_se_28, x_se_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_179 => add_58, mul_83, mul_84, sub_25
# x_183 => relu_30
# x_se_28 => mean_7
# x_se_29 => convolution_40
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 896
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


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqeqevaaxlfqvrvfl66ls2bligzo5hpts4cdsjtbl23lwdwhuxk.py
# Source Nodes: [sigmoid_7, x_184, x_185, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_7 => sigmoid_7
# x_184 => mul_85
# x_185 => convolution_42
# x_se_28 => mean_7
# x_se_29 => convolution_40
# x_se_30 => relu_31
# x_se_31 => convolution_41
triton_poi_fused_convolution_mean_mul_relu_sigmoid_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cyawuyrrgx4w2eqogz3dib5mj2ey46ebeicadgpdc4kwaduohpzh.py
# Source Nodes: [shortcut_8, x_186, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_8 => relu_32
# x_186 => add_60, mul_87, mul_88, sub_26
# x_192 => add_62, mul_90, mul_91, sub_27
# x_196 => add_63
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3mem7yginbrgojdeavidyldjdt5opy66pdm7jgzztjxsatdcrb.py
# Source Nodes: [x_200, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_200 => add_65, mul_93, mul_94, sub_28
# x_204 => relu_33
# x_205 => convolution_45
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2usznkhzewqh4wg2gdudnr7oxuwmaihiadqhgl5eqsxjfamjby.py
# Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_32 => mean_8
# x_se_33 => convolution_46
# x_se_34 => relu_35
# x_se_35 => convolution_47
triton_poi_fused_convolution_mean_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3oam6b43yjxe5ttvggcu5fievhyf523c7y52sd6myav5rpvf76n.py
# Source Nodes: [shortcut_9, x_213, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_9 => relu_36
# x_213 => add_69, mul_100, mul_101, sub_30
# x_218 => add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bffqfxd57ta44uw4rrkkdsdbfbm4swbgpj25pgzkypkaghsy3b.py
# Source Nodes: [x_421, x_425, x_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_421 => add_135, mul_193, mul_194, sub_58
# x_425 => relu_73
# x_426 => convolution_95
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1756160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2240
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbillu2gat7x4yiv4rpinmmhdzuevakhcbaxev5dgwzfotsvjo2o.py
# Source Nodes: [x_427, x_431, x_se_72, x_se_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
# x_427 => add_137, mul_196, mul_197, sub_59
# x_431 => relu_74
# x_se_72 => mean_18
# x_se_73 => convolution_96
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2240
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


# kernel path: /tmp/torchinductor_youkaichao/72/c72q3pnq2yg3zy6svbt6myhk2575w4qzvdnxzudfiitwb4cu76rl.py
# Source Nodes: [sigmoid_18, x_432, x_433, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# sigmoid_18 => sigmoid_18
# x_432 => mul_198
# x_433 => convolution_98
# x_se_72 => mean_18
# x_se_73 => convolution_96
# x_se_74 => relu_75
# x_se_75 => convolution_97
triton_poi_fused_convolution_mean_mul_relu_sigmoid_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2240
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4njylqvrs5he4wffkenfgwtfokwsezdek6awhe2sbt5mon2cwh.py
# Source Nodes: [x_434, x_440, x_444, x_447, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# x_434 => add_139, mul_200, mul_201, sub_60
# x_440 => add_141, mul_203, mul_204, sub_61
# x_444 => add_142
# x_447 => relu_76
# x_450 => mean_19
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2240
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 49.0
    tmp35 = tmp33 / tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp35, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (224, ), (1, ))
    assert_size_stride(arg3_1, (224, ), (1, ))
    assert_size_stride(arg4_1, (224, ), (1, ))
    assert_size_stride(arg5_1, (224, ), (1, ))
    assert_size_stride(arg6_1, (224, ), (1, ))
    assert_size_stride(arg7_1, (224, ), (1, ))
    assert_size_stride(arg8_1, (224, ), (1, ))
    assert_size_stride(arg9_1, (224, ), (1, ))
    assert_size_stride(arg10_1, (224, ), (1, ))
    assert_size_stride(arg11_1, (224, ), (1, ))
    assert_size_stride(arg12_1, (224, ), (1, ))
    assert_size_stride(arg13_1, (224, ), (1, ))
    assert_size_stride(arg14_1, (224, ), (1, ))
    assert_size_stride(arg15_1, (224, ), (1, ))
    assert_size_stride(arg16_1, (448, ), (1, ))
    assert_size_stride(arg17_1, (448, ), (1, ))
    assert_size_stride(arg18_1, (448, ), (1, ))
    assert_size_stride(arg19_1, (448, ), (1, ))
    assert_size_stride(arg20_1, (448, ), (1, ))
    assert_size_stride(arg21_1, (448, ), (1, ))
    assert_size_stride(arg22_1, (448, ), (1, ))
    assert_size_stride(arg23_1, (448, ), (1, ))
    assert_size_stride(arg24_1, (448, ), (1, ))
    assert_size_stride(arg25_1, (448, ), (1, ))
    assert_size_stride(arg26_1, (448, ), (1, ))
    assert_size_stride(arg27_1, (448, ), (1, ))
    assert_size_stride(arg28_1, (448, ), (1, ))
    assert_size_stride(arg29_1, (448, ), (1, ))
    assert_size_stride(arg30_1, (448, ), (1, ))
    assert_size_stride(arg31_1, (448, ), (1, ))
    assert_size_stride(arg32_1, (448, ), (1, ))
    assert_size_stride(arg33_1, (448, ), (1, ))
    assert_size_stride(arg34_1, (448, ), (1, ))
    assert_size_stride(arg35_1, (448, ), (1, ))
    assert_size_stride(arg36_1, (448, ), (1, ))
    assert_size_stride(arg37_1, (448, ), (1, ))
    assert_size_stride(arg38_1, (448, ), (1, ))
    assert_size_stride(arg39_1, (448, ), (1, ))
    assert_size_stride(arg40_1, (448, ), (1, ))
    assert_size_stride(arg41_1, (448, ), (1, ))
    assert_size_stride(arg42_1, (448, ), (1, ))
    assert_size_stride(arg43_1, (448, ), (1, ))
    assert_size_stride(arg44_1, (448, ), (1, ))
    assert_size_stride(arg45_1, (448, ), (1, ))
    assert_size_stride(arg46_1, (448, ), (1, ))
    assert_size_stride(arg47_1, (448, ), (1, ))
    assert_size_stride(arg48_1, (896, ), (1, ))
    assert_size_stride(arg49_1, (896, ), (1, ))
    assert_size_stride(arg50_1, (896, ), (1, ))
    assert_size_stride(arg51_1, (896, ), (1, ))
    assert_size_stride(arg52_1, (896, ), (1, ))
    assert_size_stride(arg53_1, (896, ), (1, ))
    assert_size_stride(arg54_1, (896, ), (1, ))
    assert_size_stride(arg55_1, (896, ), (1, ))
    assert_size_stride(arg56_1, (896, ), (1, ))
    assert_size_stride(arg57_1, (896, ), (1, ))
    assert_size_stride(arg58_1, (896, ), (1, ))
    assert_size_stride(arg59_1, (896, ), (1, ))
    assert_size_stride(arg60_1, (896, ), (1, ))
    assert_size_stride(arg61_1, (896, ), (1, ))
    assert_size_stride(arg62_1, (896, ), (1, ))
    assert_size_stride(arg63_1, (896, ), (1, ))
    assert_size_stride(arg64_1, (896, ), (1, ))
    assert_size_stride(arg65_1, (896, ), (1, ))
    assert_size_stride(arg66_1, (896, ), (1, ))
    assert_size_stride(arg67_1, (896, ), (1, ))
    assert_size_stride(arg68_1, (896, ), (1, ))
    assert_size_stride(arg69_1, (896, ), (1, ))
    assert_size_stride(arg70_1, (896, ), (1, ))
    assert_size_stride(arg71_1, (896, ), (1, ))
    assert_size_stride(arg72_1, (896, ), (1, ))
    assert_size_stride(arg73_1, (896, ), (1, ))
    assert_size_stride(arg74_1, (896, ), (1, ))
    assert_size_stride(arg75_1, (896, ), (1, ))
    assert_size_stride(arg76_1, (896, ), (1, ))
    assert_size_stride(arg77_1, (896, ), (1, ))
    assert_size_stride(arg78_1, (896, ), (1, ))
    assert_size_stride(arg79_1, (896, ), (1, ))
    assert_size_stride(arg80_1, (896, ), (1, ))
    assert_size_stride(arg81_1, (896, ), (1, ))
    assert_size_stride(arg82_1, (896, ), (1, ))
    assert_size_stride(arg83_1, (896, ), (1, ))
    assert_size_stride(arg84_1, (896, ), (1, ))
    assert_size_stride(arg85_1, (896, ), (1, ))
    assert_size_stride(arg86_1, (896, ), (1, ))
    assert_size_stride(arg87_1, (896, ), (1, ))
    assert_size_stride(arg88_1, (896, ), (1, ))
    assert_size_stride(arg89_1, (896, ), (1, ))
    assert_size_stride(arg90_1, (896, ), (1, ))
    assert_size_stride(arg91_1, (896, ), (1, ))
    assert_size_stride(arg92_1, (896, ), (1, ))
    assert_size_stride(arg93_1, (896, ), (1, ))
    assert_size_stride(arg94_1, (896, ), (1, ))
    assert_size_stride(arg95_1, (896, ), (1, ))
    assert_size_stride(arg96_1, (896, ), (1, ))
    assert_size_stride(arg97_1, (896, ), (1, ))
    assert_size_stride(arg98_1, (896, ), (1, ))
    assert_size_stride(arg99_1, (896, ), (1, ))
    assert_size_stride(arg100_1, (896, ), (1, ))
    assert_size_stride(arg101_1, (896, ), (1, ))
    assert_size_stride(arg102_1, (896, ), (1, ))
    assert_size_stride(arg103_1, (896, ), (1, ))
    assert_size_stride(arg104_1, (896, ), (1, ))
    assert_size_stride(arg105_1, (896, ), (1, ))
    assert_size_stride(arg106_1, (896, ), (1, ))
    assert_size_stride(arg107_1, (896, ), (1, ))
    assert_size_stride(arg108_1, (896, ), (1, ))
    assert_size_stride(arg109_1, (896, ), (1, ))
    assert_size_stride(arg110_1, (896, ), (1, ))
    assert_size_stride(arg111_1, (896, ), (1, ))
    assert_size_stride(arg112_1, (896, ), (1, ))
    assert_size_stride(arg113_1, (896, ), (1, ))
    assert_size_stride(arg114_1, (896, ), (1, ))
    assert_size_stride(arg115_1, (896, ), (1, ))
    assert_size_stride(arg116_1, (2240, ), (1, ))
    assert_size_stride(arg117_1, (2240, ), (1, ))
    assert_size_stride(arg118_1, (2240, ), (1, ))
    assert_size_stride(arg119_1, (2240, ), (1, ))
    assert_size_stride(arg120_1, (2240, ), (1, ))
    assert_size_stride(arg121_1, (2240, ), (1, ))
    assert_size_stride(arg122_1, (2240, ), (1, ))
    assert_size_stride(arg123_1, (2240, ), (1, ))
    assert_size_stride(arg124_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg125_1, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg126_1, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg127_1, (8, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg128_1, (8, ), (1, ))
    assert_size_stride(arg129_1, (224, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg130_1, (224, ), (1, ))
    assert_size_stride(arg131_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg132_1, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg133_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg134_1, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg135_1, (56, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg136_1, (56, ), (1, ))
    assert_size_stride(arg137_1, (224, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg138_1, (224, ), (1, ))
    assert_size_stride(arg139_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg140_1, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg141_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg142_1, (56, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg143_1, (56, ), (1, ))
    assert_size_stride(arg144_1, (448, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg145_1, (448, ), (1, ))
    assert_size_stride(arg146_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg147_1, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg148_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg149_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg150_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg151_1, (112, ), (1, ))
    assert_size_stride(arg152_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg153_1, (448, ), (1, ))
    assert_size_stride(arg154_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg155_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg156_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg157_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg158_1, (112, ), (1, ))
    assert_size_stride(arg159_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg160_1, (448, ), (1, ))
    assert_size_stride(arg161_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg162_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg163_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg164_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg165_1, (112, ), (1, ))
    assert_size_stride(arg166_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg167_1, (448, ), (1, ))
    assert_size_stride(arg168_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg169_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg170_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg171_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg172_1, (112, ), (1, ))
    assert_size_stride(arg173_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg174_1, (448, ), (1, ))
    assert_size_stride(arg175_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg176_1, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg177_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg178_1, (112, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg179_1, (112, ), (1, ))
    assert_size_stride(arg180_1, (896, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg181_1, (896, ), (1, ))
    assert_size_stride(arg182_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg183_1, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg184_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg185_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg186_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg187_1, (224, ), (1, ))
    assert_size_stride(arg188_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg189_1, (896, ), (1, ))
    assert_size_stride(arg190_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg191_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg192_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg193_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg194_1, (224, ), (1, ))
    assert_size_stride(arg195_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg196_1, (896, ), (1, ))
    assert_size_stride(arg197_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg198_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg199_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg200_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg201_1, (224, ), (1, ))
    assert_size_stride(arg202_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg203_1, (896, ), (1, ))
    assert_size_stride(arg204_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg205_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg206_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg207_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg208_1, (224, ), (1, ))
    assert_size_stride(arg209_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg210_1, (896, ), (1, ))
    assert_size_stride(arg211_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg212_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg213_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg214_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg215_1, (224, ), (1, ))
    assert_size_stride(arg216_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg217_1, (896, ), (1, ))
    assert_size_stride(arg218_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg219_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg220_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg221_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg222_1, (224, ), (1, ))
    assert_size_stride(arg223_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg224_1, (896, ), (1, ))
    assert_size_stride(arg225_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg226_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg227_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg228_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg229_1, (224, ), (1, ))
    assert_size_stride(arg230_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg231_1, (896, ), (1, ))
    assert_size_stride(arg232_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg233_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg234_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg235_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg236_1, (224, ), (1, ))
    assert_size_stride(arg237_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg238_1, (896, ), (1, ))
    assert_size_stride(arg239_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg240_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg241_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg242_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg243_1, (224, ), (1, ))
    assert_size_stride(arg244_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg245_1, (896, ), (1, ))
    assert_size_stride(arg246_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg247_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg248_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg249_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg250_1, (224, ), (1, ))
    assert_size_stride(arg251_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg252_1, (896, ), (1, ))
    assert_size_stride(arg253_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg254_1, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg255_1, (2240, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg256_1, (224, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg257_1, (224, ), (1, ))
    assert_size_stride(arg258_1, (2240, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg259_1, (2240, ), (1, ))
    assert_size_stride(arg260_1, (2240, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg261_1, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg262_1, (1000, 2240), (2240, 1))
    assert_size_stride(arg263_1, (1000, ), (1, ))
    assert_size_stride(arg264_1, (32, ), (1, ))
    assert_size_stride(arg265_1, (32, ), (1, ))
    assert_size_stride(arg266_1, (224, ), (1, ))
    assert_size_stride(arg267_1, (224, ), (1, ))
    assert_size_stride(arg268_1, (224, ), (1, ))
    assert_size_stride(arg269_1, (224, ), (1, ))
    assert_size_stride(arg270_1, (224, ), (1, ))
    assert_size_stride(arg271_1, (224, ), (1, ))
    assert_size_stride(arg272_1, (224, ), (1, ))
    assert_size_stride(arg273_1, (224, ), (1, ))
    assert_size_stride(arg274_1, (224, ), (1, ))
    assert_size_stride(arg275_1, (224, ), (1, ))
    assert_size_stride(arg276_1, (224, ), (1, ))
    assert_size_stride(arg277_1, (224, ), (1, ))
    assert_size_stride(arg278_1, (224, ), (1, ))
    assert_size_stride(arg279_1, (224, ), (1, ))
    assert_size_stride(arg280_1, (448, ), (1, ))
    assert_size_stride(arg281_1, (448, ), (1, ))
    assert_size_stride(arg282_1, (448, ), (1, ))
    assert_size_stride(arg283_1, (448, ), (1, ))
    assert_size_stride(arg284_1, (448, ), (1, ))
    assert_size_stride(arg285_1, (448, ), (1, ))
    assert_size_stride(arg286_1, (448, ), (1, ))
    assert_size_stride(arg287_1, (448, ), (1, ))
    assert_size_stride(arg288_1, (448, ), (1, ))
    assert_size_stride(arg289_1, (448, ), (1, ))
    assert_size_stride(arg290_1, (448, ), (1, ))
    assert_size_stride(arg291_1, (448, ), (1, ))
    assert_size_stride(arg292_1, (448, ), (1, ))
    assert_size_stride(arg293_1, (448, ), (1, ))
    assert_size_stride(arg294_1, (448, ), (1, ))
    assert_size_stride(arg295_1, (448, ), (1, ))
    assert_size_stride(arg296_1, (448, ), (1, ))
    assert_size_stride(arg297_1, (448, ), (1, ))
    assert_size_stride(arg298_1, (448, ), (1, ))
    assert_size_stride(arg299_1, (448, ), (1, ))
    assert_size_stride(arg300_1, (448, ), (1, ))
    assert_size_stride(arg301_1, (448, ), (1, ))
    assert_size_stride(arg302_1, (448, ), (1, ))
    assert_size_stride(arg303_1, (448, ), (1, ))
    assert_size_stride(arg304_1, (448, ), (1, ))
    assert_size_stride(arg305_1, (448, ), (1, ))
    assert_size_stride(arg306_1, (448, ), (1, ))
    assert_size_stride(arg307_1, (448, ), (1, ))
    assert_size_stride(arg308_1, (448, ), (1, ))
    assert_size_stride(arg309_1, (448, ), (1, ))
    assert_size_stride(arg310_1, (448, ), (1, ))
    assert_size_stride(arg311_1, (448, ), (1, ))
    assert_size_stride(arg312_1, (896, ), (1, ))
    assert_size_stride(arg313_1, (896, ), (1, ))
    assert_size_stride(arg314_1, (896, ), (1, ))
    assert_size_stride(arg315_1, (896, ), (1, ))
    assert_size_stride(arg316_1, (896, ), (1, ))
    assert_size_stride(arg317_1, (896, ), (1, ))
    assert_size_stride(arg318_1, (896, ), (1, ))
    assert_size_stride(arg319_1, (896, ), (1, ))
    assert_size_stride(arg320_1, (896, ), (1, ))
    assert_size_stride(arg321_1, (896, ), (1, ))
    assert_size_stride(arg322_1, (896, ), (1, ))
    assert_size_stride(arg323_1, (896, ), (1, ))
    assert_size_stride(arg324_1, (896, ), (1, ))
    assert_size_stride(arg325_1, (896, ), (1, ))
    assert_size_stride(arg326_1, (896, ), (1, ))
    assert_size_stride(arg327_1, (896, ), (1, ))
    assert_size_stride(arg328_1, (896, ), (1, ))
    assert_size_stride(arg329_1, (896, ), (1, ))
    assert_size_stride(arg330_1, (896, ), (1, ))
    assert_size_stride(arg331_1, (896, ), (1, ))
    assert_size_stride(arg332_1, (896, ), (1, ))
    assert_size_stride(arg333_1, (896, ), (1, ))
    assert_size_stride(arg334_1, (896, ), (1, ))
    assert_size_stride(arg335_1, (896, ), (1, ))
    assert_size_stride(arg336_1, (896, ), (1, ))
    assert_size_stride(arg337_1, (896, ), (1, ))
    assert_size_stride(arg338_1, (896, ), (1, ))
    assert_size_stride(arg339_1, (896, ), (1, ))
    assert_size_stride(arg340_1, (896, ), (1, ))
    assert_size_stride(arg341_1, (896, ), (1, ))
    assert_size_stride(arg342_1, (896, ), (1, ))
    assert_size_stride(arg343_1, (896, ), (1, ))
    assert_size_stride(arg344_1, (896, ), (1, ))
    assert_size_stride(arg345_1, (896, ), (1, ))
    assert_size_stride(arg346_1, (896, ), (1, ))
    assert_size_stride(arg347_1, (896, ), (1, ))
    assert_size_stride(arg348_1, (896, ), (1, ))
    assert_size_stride(arg349_1, (896, ), (1, ))
    assert_size_stride(arg350_1, (896, ), (1, ))
    assert_size_stride(arg351_1, (896, ), (1, ))
    assert_size_stride(arg352_1, (896, ), (1, ))
    assert_size_stride(arg353_1, (896, ), (1, ))
    assert_size_stride(arg354_1, (896, ), (1, ))
    assert_size_stride(arg355_1, (896, ), (1, ))
    assert_size_stride(arg356_1, (896, ), (1, ))
    assert_size_stride(arg357_1, (896, ), (1, ))
    assert_size_stride(arg358_1, (896, ), (1, ))
    assert_size_stride(arg359_1, (896, ), (1, ))
    assert_size_stride(arg360_1, (896, ), (1, ))
    assert_size_stride(arg361_1, (896, ), (1, ))
    assert_size_stride(arg362_1, (896, ), (1, ))
    assert_size_stride(arg363_1, (896, ), (1, ))
    assert_size_stride(arg364_1, (896, ), (1, ))
    assert_size_stride(arg365_1, (896, ), (1, ))
    assert_size_stride(arg366_1, (896, ), (1, ))
    assert_size_stride(arg367_1, (896, ), (1, ))
    assert_size_stride(arg368_1, (896, ), (1, ))
    assert_size_stride(arg369_1, (896, ), (1, ))
    assert_size_stride(arg370_1, (896, ), (1, ))
    assert_size_stride(arg371_1, (896, ), (1, ))
    assert_size_stride(arg372_1, (896, ), (1, ))
    assert_size_stride(arg373_1, (896, ), (1, ))
    assert_size_stride(arg374_1, (896, ), (1, ))
    assert_size_stride(arg375_1, (896, ), (1, ))
    assert_size_stride(arg376_1, (896, ), (1, ))
    assert_size_stride(arg377_1, (896, ), (1, ))
    assert_size_stride(arg378_1, (896, ), (1, ))
    assert_size_stride(arg379_1, (896, ), (1, ))
    assert_size_stride(arg380_1, (2240, ), (1, ))
    assert_size_stride(arg381_1, (2240, ), (1, ))
    assert_size_stride(arg382_1, (2240, ), (1, ))
    assert_size_stride(arg383_1, (2240, ), (1, ))
    assert_size_stride(arg384_1, (2240, ), (1, ))
    assert_size_stride(arg385_1, (2240, ), (1, ))
    assert_size_stride(arg386_1, (2240, ), (1, ))
    assert_size_stride(arg387_1, (2240, ), (1, ))
    assert_size_stride(arg388_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg388_1, arg124_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 112, 112), (401408, 12544, 112, 1))
        del arg124_1
        del arg388_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg264_1, arg265_1, arg0_1, arg1_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg0_1
        del arg1_1
        del arg264_1
        del arg265_1
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 224, 112, 112), (2809856, 12544, 112, 1))
        del arg125_1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf3, arg266_1, arg267_1, arg2_1, arg3_1, 11239424, grid=grid(11239424), stream=stream0)
        del arg266_1
        del arg267_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg126_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf4, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg126_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf6, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf6  # reuse
        # Source Nodes: [x_13, x_17, x_se, x_se_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2.run(buf5, buf7, arg268_1, arg269_1, arg4_1, arg5_1, 896, 3136, grid=grid(896), stream=stream0)
        del arg268_1
        del arg269_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf8 = extern_kernels.convolution(buf7, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 8, 1, 1), (8, 1, 1, 1))
        del arg127_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_3.run(buf9, arg128_1, 32, grid=grid(32), stream=stream0)
        del arg128_1
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg129_1
        del buf9
        buf11 = buf5; del buf5  # reuse
        # Source Nodes: [sigmoid, x_18, x_19, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_4.run(buf11, buf10, arg130_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg130_1
        # Source Nodes: [sigmoid, x_18, x_19, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf12 = extern_kernels.convolution(buf11, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg131_1
        del buf11
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf1, arg132_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg132_1
        del buf1
        buf14 = buf12; del buf12  # reuse
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf15, arg270_1, arg271_1, arg6_1, arg7_1, buf13, arg272_1, arg273_1, arg8_1, arg9_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg270_1
        del arg271_1
        del arg272_1
        del arg273_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf13
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg133_1
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [x_34, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf17, arg274_1, arg275_1, arg10_1, arg11_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg10_1
        del arg11_1
        del arg274_1
        del arg275_1
        # Source Nodes: [x_34, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg134_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf18, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg134_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = reinterpret_tensor(buf10, (4, 224, 1, 1), (224, 1, 896, 896), 0); del buf10  # reuse
        buf21 = reinterpret_tensor(buf20, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf20  # reuse
        # Source Nodes: [x_40, x_44, x_se_4, x_se_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_2.run(buf19, buf21, arg276_1, arg277_1, arg12_1, arg13_1, 896, 3136, grid=grid(896), stream=stream0)
        del arg12_1
        del arg13_1
        del arg276_1
        del arg277_1
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf22 = extern_kernels.convolution(buf21, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 56, 1, 1), (56, 1, 1, 1))
        del arg135_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_7.run(buf23, arg136_1, 224, grid=grid(224), stream=stream0)
        del arg136_1
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg137_1
        del buf23
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [sigmoid_1, x_45, x_46, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_4.run(buf25, buf24, arg138_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg138_1
        del buf24
        # Source Nodes: [sigmoid_1, x_45, x_46, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf26 = extern_kernels.convolution(buf25, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 224, 56, 56), (702464, 3136, 56, 1))
        del arg139_1
        del buf25
        buf27 = buf15; del buf15  # reuse
        # Source Nodes: [shortcut_2, x_47, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf27, buf26, arg278_1, arg279_1, arg14_1, arg15_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg14_1
        del arg15_1
        del arg278_1
        del arg279_1
        del buf26
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 448, 56, 56), (1404928, 3136, 56, 1))
        del arg140_1
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [x_57, x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf29, arg280_1, arg281_1, arg16_1, arg17_1, 5619712, grid=grid(5619712), stream=stream0)
        del arg16_1
        del arg17_1
        del arg280_1
        del arg281_1
        # Source Nodes: [x_57, x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg141_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf30, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg141_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf33 = reinterpret_tensor(buf32, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf32  # reuse
        # Source Nodes: [x_63, x_67, x_se_8, x_se_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10.run(buf31, buf33, arg282_1, arg283_1, arg18_1, arg19_1, 1792, 784, grid=grid(1792), stream=stream0)
        del arg18_1
        del arg19_1
        del arg282_1
        del arg283_1
        # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
        buf34 = extern_kernels.convolution(buf33, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 56, 1, 1), (56, 1, 1, 1))
        del arg142_1
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_7.run(buf35, arg143_1, 224, grid=grid(224), stream=stream0)
        del arg143_1
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 448, 1, 1), (448, 1, 1, 1))
        del arg144_1
        del buf35
        buf37 = buf31; del buf31  # reuse
        # Source Nodes: [sigmoid_2, x_68, x_69, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_11.run(buf37, buf36, arg145_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg145_1
        # Source Nodes: [sigmoid_2, x_68, x_69, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf38 = extern_kernels.convolution(buf37, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg146_1
        del buf37
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf27, arg147_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg147_1
        del buf27
        buf40 = buf38; del buf38  # reuse
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [shortcut_3, x_70, x_76, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf41, arg284_1, arg285_1, arg20_1, arg21_1, buf39, arg286_1, arg287_1, arg22_1, arg23_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        del arg284_1
        del arg285_1
        del arg286_1
        del arg287_1
        del buf39
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg148_1
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [x_84, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf43, arg288_1, arg289_1, arg24_1, arg25_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg24_1
        del arg25_1
        del arg288_1
        del arg289_1
        # Source Nodes: [x_84, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf44 = extern_kernels.convolution(buf43, arg149_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf44, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg149_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        buf46 = reinterpret_tensor(buf36, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf36  # reuse
        buf47 = reinterpret_tensor(buf46, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf46  # reuse
        # Source Nodes: [x_90, x_94, x_se_12, x_se_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10.run(buf45, buf47, arg290_1, arg291_1, arg26_1, arg27_1, 1792, 784, grid=grid(1792), stream=stream0)
        del arg26_1
        del arg27_1
        del arg290_1
        del arg291_1
        # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
        buf48 = extern_kernels.convolution(buf47, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 112, 1, 1), (112, 1, 1, 1))
        del arg150_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_14.run(buf49, arg151_1, 448, grid=grid(448), stream=stream0)
        del arg151_1
        # Source Nodes: [x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf50 = extern_kernels.convolution(buf49, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 448, 1, 1), (448, 1, 1, 1))
        del arg152_1
        del buf49
        buf51 = buf45; del buf45  # reuse
        # Source Nodes: [sigmoid_3, x_95, x_96, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_11.run(buf51, buf50, arg153_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg153_1
        # Source Nodes: [sigmoid_3, x_95, x_96, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf52 = extern_kernels.convolution(buf51, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg154_1
        del buf51
        buf53 = buf41; del buf41  # reuse
        # Source Nodes: [shortcut_4, x_102, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf53, buf52, arg292_1, arg293_1, arg28_1, arg29_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg28_1
        del arg292_1
        del arg293_1
        del arg29_1
        del buf52
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg155_1
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [x_106, x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf55, arg294_1, arg295_1, arg30_1, arg31_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg294_1
        del arg295_1
        del arg30_1
        del arg31_1
        # Source Nodes: [x_106, x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf56, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg156_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        buf58 = reinterpret_tensor(buf50, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf50  # reuse
        buf59 = reinterpret_tensor(buf58, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf58  # reuse
        # Source Nodes: [x_112, x_116, x_se_16, x_se_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10.run(buf57, buf59, arg296_1, arg297_1, arg32_1, arg33_1, 1792, 784, grid=grid(1792), stream=stream0)
        del arg296_1
        del arg297_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
        buf60 = extern_kernels.convolution(buf59, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 112, 1, 1), (112, 1, 1, 1))
        del arg157_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_14.run(buf61, arg158_1, 448, grid=grid(448), stream=stream0)
        del arg158_1
        # Source Nodes: [x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 448, 1, 1), (448, 1, 1, 1))
        del arg159_1
        del buf61
        buf63 = buf57; del buf57  # reuse
        # Source Nodes: [sigmoid_4, x_117, x_118, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_11.run(buf63, buf62, arg160_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg160_1
        # Source Nodes: [sigmoid_4, x_117, x_118, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf64 = extern_kernels.convolution(buf63, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg161_1
        del buf63
        buf65 = buf53; del buf53  # reuse
        # Source Nodes: [shortcut_5, x_119, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf65, buf64, arg298_1, arg299_1, arg34_1, arg35_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg298_1
        del arg299_1
        del arg34_1
        del arg35_1
        del buf64
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg162_1
        buf67 = buf66; del buf66  # reuse
        # Source Nodes: [x_128, x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf67, arg300_1, arg301_1, arg36_1, arg37_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg300_1
        del arg301_1
        del arg36_1
        del arg37_1
        # Source Nodes: [x_128, x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf67, arg163_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf68, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg163_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        buf70 = reinterpret_tensor(buf62, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf62  # reuse
        buf71 = reinterpret_tensor(buf70, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf70  # reuse
        # Source Nodes: [x_134, x_138, x_se_20, x_se_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10.run(buf69, buf71, arg302_1, arg303_1, arg38_1, arg39_1, 1792, 784, grid=grid(1792), stream=stream0)
        del arg302_1
        del arg303_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
        buf72 = extern_kernels.convolution(buf71, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 112, 1, 1), (112, 1, 1, 1))
        del arg164_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Source Nodes: [x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_14.run(buf73, arg165_1, 448, grid=grid(448), stream=stream0)
        del arg165_1
        # Source Nodes: [x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 448, 1, 1), (448, 1, 1, 1))
        del arg166_1
        del buf73
        buf75 = buf69; del buf69  # reuse
        # Source Nodes: [sigmoid_5, x_139, x_140, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_11.run(buf75, buf74, arg167_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg167_1
        # Source Nodes: [sigmoid_5, x_139, x_140, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf76 = extern_kernels.convolution(buf75, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg168_1
        del buf75
        buf77 = buf65; del buf65  # reuse
        # Source Nodes: [shortcut_6, x_141, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf77, buf76, arg304_1, arg305_1, arg40_1, arg41_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg304_1
        del arg305_1
        del arg40_1
        del arg41_1
        del buf76
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg169_1
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [x_150, x_154, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf79, arg306_1, arg307_1, arg42_1, arg43_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg306_1
        del arg307_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_150, x_154, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf80, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg170_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        buf82 = reinterpret_tensor(buf74, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf74  # reuse
        buf83 = reinterpret_tensor(buf82, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf82  # reuse
        # Source Nodes: [x_156, x_160, x_se_24, x_se_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_10.run(buf81, buf83, arg308_1, arg309_1, arg44_1, arg45_1, 1792, 784, grid=grid(1792), stream=stream0)
        del arg308_1
        del arg309_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
        buf84 = extern_kernels.convolution(buf83, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 112, 1, 1), (112, 1, 1, 1))
        del arg171_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Source Nodes: [x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_14.run(buf85, arg172_1, 448, grid=grid(448), stream=stream0)
        del arg172_1
        # Source Nodes: [x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 448, 1, 1), (448, 1, 1, 1))
        del arg173_1
        del buf85
        buf87 = buf81; del buf81  # reuse
        # Source Nodes: [sigmoid_6, x_161, x_162, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_11.run(buf87, buf86, arg174_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg174_1
        del buf86
        # Source Nodes: [sigmoid_6, x_161, x_162, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf88 = extern_kernels.convolution(buf87, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 448, 28, 28), (351232, 784, 28, 1))
        del arg175_1
        del buf87
        buf89 = buf77; del buf77  # reuse
        # Source Nodes: [shortcut_7, x_163, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf89, buf88, arg310_1, arg311_1, arg46_1, arg47_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg310_1
        del arg311_1
        del arg46_1
        del arg47_1
        del buf88
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 896, 28, 28), (702464, 784, 28, 1))
        del arg176_1
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [x_173, x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf91, arg312_1, arg313_1, arg48_1, arg49_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg312_1
        del arg313_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_173, x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf91, arg177_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf92, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg177_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf95 = reinterpret_tensor(buf94, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf94  # reuse
        # Source Nodes: [x_179, x_183, x_se_28, x_se_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf93, buf95, arg314_1, arg315_1, arg50_1, arg51_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg314_1
        del arg315_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
        buf96 = extern_kernels.convolution(buf95, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 112, 1, 1), (112, 1, 1, 1))
        del arg178_1
        del buf95
        buf97 = buf96; del buf96  # reuse
        # Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_14.run(buf97, arg179_1, 448, grid=grid(448), stream=stream0)
        del arg179_1
        # Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg180_1
        del buf97
        buf99 = buf93; del buf93  # reuse
        # Source Nodes: [sigmoid_7, x_184, x_185, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf99, buf98, arg181_1, 702464, grid=grid(702464), stream=stream0)
        del arg181_1
        # Source Nodes: [sigmoid_7, x_184, x_185, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf100 = extern_kernels.convolution(buf99, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg182_1
        del buf99
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf89, arg183_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg183_1
        del buf89
        buf102 = buf100; del buf100  # reuse
        buf103 = buf102; del buf102  # reuse
        # Source Nodes: [shortcut_8, x_186, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf103, arg316_1, arg317_1, arg52_1, arg53_1, buf101, arg318_1, arg319_1, arg54_1, arg55_1, 702464, grid=grid(702464), stream=stream0)
        del arg316_1
        del arg317_1
        del arg318_1
        del arg319_1
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf101
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg184_1
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [x_200, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf105, arg320_1, arg321_1, arg56_1, arg57_1, 702464, grid=grid(702464), stream=stream0)
        del arg320_1
        del arg321_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_200, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf106 = extern_kernels.convolution(buf105, arg185_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf106, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg185_1
        del buf105
        buf107 = buf106; del buf106  # reuse
        buf108 = reinterpret_tensor(buf98, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf98  # reuse
        buf109 = reinterpret_tensor(buf108, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf108  # reuse
        # Source Nodes: [x_206, x_210, x_se_32, x_se_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf107, buf109, arg322_1, arg323_1, arg58_1, arg59_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg322_1
        del arg323_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
        buf110 = extern_kernels.convolution(buf109, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg186_1
        del buf109
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf111, arg187_1, 896, grid=grid(896), stream=stream0)
        del arg187_1
        # Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg188_1
        del buf111
        buf113 = buf107; del buf107  # reuse
        # Source Nodes: [sigmoid_8, x_211, x_212, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf113, buf112, arg189_1, 702464, grid=grid(702464), stream=stream0)
        del arg189_1
        # Source Nodes: [sigmoid_8, x_211, x_212, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf114 = extern_kernels.convolution(buf113, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg190_1
        del buf113
        buf115 = buf103; del buf103  # reuse
        # Source Nodes: [shortcut_9, x_213, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf115, buf114, arg324_1, arg325_1, arg60_1, arg61_1, 702464, grid=grid(702464), stream=stream0)
        del arg324_1
        del arg325_1
        del arg60_1
        del arg61_1
        del buf114
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg191_1
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [x_222, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf117, arg326_1, arg327_1, arg62_1, arg63_1, 702464, grid=grid(702464), stream=stream0)
        del arg326_1
        del arg327_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_222, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf118 = extern_kernels.convolution(buf117, arg192_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf118, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg192_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        buf120 = reinterpret_tensor(buf112, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf112  # reuse
        buf121 = reinterpret_tensor(buf120, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf120  # reuse
        # Source Nodes: [x_228, x_232, x_se_36, x_se_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf119, buf121, arg328_1, arg329_1, arg64_1, arg65_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg328_1
        del arg329_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
        buf122 = extern_kernels.convolution(buf121, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg193_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf123, arg194_1, 896, grid=grid(896), stream=stream0)
        del arg194_1
        # Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf124 = extern_kernels.convolution(buf123, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg195_1
        del buf123
        buf125 = buf119; del buf119  # reuse
        # Source Nodes: [sigmoid_9, x_233, x_234, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf125, buf124, arg196_1, 702464, grid=grid(702464), stream=stream0)
        del arg196_1
        # Source Nodes: [sigmoid_9, x_233, x_234, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf126 = extern_kernels.convolution(buf125, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg197_1
        del buf125
        buf127 = buf115; del buf115  # reuse
        # Source Nodes: [shortcut_10, x_235, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf127, buf126, arg330_1, arg331_1, arg66_1, arg67_1, 702464, grid=grid(702464), stream=stream0)
        del arg330_1
        del arg331_1
        del arg66_1
        del arg67_1
        del buf126
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg198_1
        buf129 = buf128; del buf128  # reuse
        # Source Nodes: [x_244, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf129, arg332_1, arg333_1, arg68_1, arg69_1, 702464, grid=grid(702464), stream=stream0)
        del arg332_1
        del arg333_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_244, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf130 = extern_kernels.convolution(buf129, arg199_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf130, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg199_1
        del buf129
        buf131 = buf130; del buf130  # reuse
        buf132 = reinterpret_tensor(buf124, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf124  # reuse
        buf133 = reinterpret_tensor(buf132, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf132  # reuse
        # Source Nodes: [x_250, x_254, x_se_40, x_se_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf131, buf133, arg334_1, arg335_1, arg70_1, arg71_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg334_1
        del arg335_1
        del arg70_1
        del arg71_1
        # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
        buf134 = extern_kernels.convolution(buf133, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg200_1
        del buf133
        buf135 = buf134; del buf134  # reuse
        # Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf135, arg201_1, 896, grid=grid(896), stream=stream0)
        del arg201_1
        # Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf136 = extern_kernels.convolution(buf135, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg202_1
        del buf135
        buf137 = buf131; del buf131  # reuse
        # Source Nodes: [sigmoid_10, x_255, x_256, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf137, buf136, arg203_1, 702464, grid=grid(702464), stream=stream0)
        del arg203_1
        # Source Nodes: [sigmoid_10, x_255, x_256, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf138 = extern_kernels.convolution(buf137, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg204_1
        del buf137
        buf139 = buf127; del buf127  # reuse
        # Source Nodes: [shortcut_11, x_257, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf139, buf138, arg336_1, arg337_1, arg72_1, arg73_1, 702464, grid=grid(702464), stream=stream0)
        del arg336_1
        del arg337_1
        del arg72_1
        del arg73_1
        del buf138
        # Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg205_1
        buf141 = buf140; del buf140  # reuse
        # Source Nodes: [x_266, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf141, arg338_1, arg339_1, arg74_1, arg75_1, 702464, grid=grid(702464), stream=stream0)
        del arg338_1
        del arg339_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_266, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf142 = extern_kernels.convolution(buf141, arg206_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf142, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg206_1
        del buf141
        buf143 = buf142; del buf142  # reuse
        buf144 = reinterpret_tensor(buf136, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf136  # reuse
        buf145 = reinterpret_tensor(buf144, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf144  # reuse
        # Source Nodes: [x_272, x_276, x_se_44, x_se_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf143, buf145, arg340_1, arg341_1, arg76_1, arg77_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg340_1
        del arg341_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
        buf146 = extern_kernels.convolution(buf145, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg207_1
        del buf145
        buf147 = buf146; del buf146  # reuse
        # Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf147, arg208_1, 896, grid=grid(896), stream=stream0)
        del arg208_1
        # Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf148 = extern_kernels.convolution(buf147, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg209_1
        del buf147
        buf149 = buf143; del buf143  # reuse
        # Source Nodes: [sigmoid_11, x_277, x_278, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf149, buf148, arg210_1, 702464, grid=grid(702464), stream=stream0)
        del arg210_1
        # Source Nodes: [sigmoid_11, x_277, x_278, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf150 = extern_kernels.convolution(buf149, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg211_1
        del buf149
        buf151 = buf139; del buf139  # reuse
        # Source Nodes: [shortcut_12, x_279, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf151, buf150, arg342_1, arg343_1, arg78_1, arg79_1, 702464, grid=grid(702464), stream=stream0)
        del arg342_1
        del arg343_1
        del arg78_1
        del arg79_1
        del buf150
        # Source Nodes: [x_287], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg212_1
        buf153 = buf152; del buf152  # reuse
        # Source Nodes: [x_288, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf153, arg344_1, arg345_1, arg80_1, arg81_1, 702464, grid=grid(702464), stream=stream0)
        del arg344_1
        del arg345_1
        del arg80_1
        del arg81_1
        # Source Nodes: [x_288, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf154 = extern_kernels.convolution(buf153, arg213_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf154, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg213_1
        del buf153
        buf155 = buf154; del buf154  # reuse
        buf156 = reinterpret_tensor(buf148, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf148  # reuse
        buf157 = reinterpret_tensor(buf156, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf156  # reuse
        # Source Nodes: [x_294, x_298, x_se_48, x_se_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf155, buf157, arg346_1, arg347_1, arg82_1, arg83_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg346_1
        del arg347_1
        del arg82_1
        del arg83_1
        # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
        buf158 = extern_kernels.convolution(buf157, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg214_1
        del buf157
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf159, arg215_1, 896, grid=grid(896), stream=stream0)
        del arg215_1
        # Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf160 = extern_kernels.convolution(buf159, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg216_1
        del buf159
        buf161 = buf155; del buf155  # reuse
        # Source Nodes: [sigmoid_12, x_299, x_300, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf161, buf160, arg217_1, 702464, grid=grid(702464), stream=stream0)
        del arg217_1
        # Source Nodes: [sigmoid_12, x_299, x_300, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf162 = extern_kernels.convolution(buf161, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg218_1
        del buf161
        buf163 = buf151; del buf151  # reuse
        # Source Nodes: [shortcut_13, x_301, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf163, buf162, arg348_1, arg349_1, arg84_1, arg85_1, 702464, grid=grid(702464), stream=stream0)
        del arg348_1
        del arg349_1
        del arg84_1
        del arg85_1
        del buf162
        # Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg219_1
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [x_310, x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf165, arg350_1, arg351_1, arg86_1, arg87_1, 702464, grid=grid(702464), stream=stream0)
        del arg350_1
        del arg351_1
        del arg86_1
        del arg87_1
        # Source Nodes: [x_310, x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf166 = extern_kernels.convolution(buf165, arg220_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf166, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg220_1
        del buf165
        buf167 = buf166; del buf166  # reuse
        buf168 = reinterpret_tensor(buf160, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf160  # reuse
        buf169 = reinterpret_tensor(buf168, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf168  # reuse
        # Source Nodes: [x_316, x_320, x_se_52, x_se_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf167, buf169, arg352_1, arg353_1, arg88_1, arg89_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg352_1
        del arg353_1
        del arg88_1
        del arg89_1
        # Source Nodes: [x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean]
        buf170 = extern_kernels.convolution(buf169, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg221_1
        del buf169
        buf171 = buf170; del buf170  # reuse
        # Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf171, arg222_1, 896, grid=grid(896), stream=stream0)
        del arg222_1
        # Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf172 = extern_kernels.convolution(buf171, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg223_1
        del buf171
        buf173 = buf167; del buf167  # reuse
        # Source Nodes: [sigmoid_13, x_321, x_322, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf173, buf172, arg224_1, 702464, grid=grid(702464), stream=stream0)
        del arg224_1
        # Source Nodes: [sigmoid_13, x_321, x_322, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf174 = extern_kernels.convolution(buf173, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg225_1
        del buf173
        buf175 = buf163; del buf163  # reuse
        # Source Nodes: [shortcut_14, x_323, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf175, buf174, arg354_1, arg355_1, arg90_1, arg91_1, 702464, grid=grid(702464), stream=stream0)
        del arg354_1
        del arg355_1
        del arg90_1
        del arg91_1
        del buf174
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg226_1
        buf177 = buf176; del buf176  # reuse
        # Source Nodes: [x_332, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf177, arg356_1, arg357_1, arg92_1, arg93_1, 702464, grid=grid(702464), stream=stream0)
        del arg356_1
        del arg357_1
        del arg92_1
        del arg93_1
        # Source Nodes: [x_332, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf178 = extern_kernels.convolution(buf177, arg227_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf178, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg227_1
        del buf177
        buf179 = buf178; del buf178  # reuse
        buf180 = reinterpret_tensor(buf172, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf172  # reuse
        buf181 = reinterpret_tensor(buf180, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf180  # reuse
        # Source Nodes: [x_338, x_342, x_se_56, x_se_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf179, buf181, arg358_1, arg359_1, arg94_1, arg95_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg358_1
        del arg359_1
        del arg94_1
        del arg95_1
        # Source Nodes: [x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean]
        buf182 = extern_kernels.convolution(buf181, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg228_1
        del buf181
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf183, arg229_1, 896, grid=grid(896), stream=stream0)
        del arg229_1
        # Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf184 = extern_kernels.convolution(buf183, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg230_1
        del buf183
        buf185 = buf179; del buf179  # reuse
        # Source Nodes: [sigmoid_14, x_343, x_344, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf185, buf184, arg231_1, 702464, grid=grid(702464), stream=stream0)
        del arg231_1
        # Source Nodes: [sigmoid_14, x_343, x_344, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf186 = extern_kernels.convolution(buf185, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg232_1
        del buf185
        buf187 = buf175; del buf175  # reuse
        # Source Nodes: [shortcut_15, x_345, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf187, buf186, arg360_1, arg361_1, arg96_1, arg97_1, 702464, grid=grid(702464), stream=stream0)
        del arg360_1
        del arg361_1
        del arg96_1
        del arg97_1
        del buf186
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg233_1
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [x_354, x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf189, arg362_1, arg363_1, arg98_1, arg99_1, 702464, grid=grid(702464), stream=stream0)
        del arg362_1
        del arg363_1
        del arg98_1
        del arg99_1
        # Source Nodes: [x_354, x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf190 = extern_kernels.convolution(buf189, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf190, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg234_1
        del buf189
        buf191 = buf190; del buf190  # reuse
        buf192 = reinterpret_tensor(buf184, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf184  # reuse
        buf193 = reinterpret_tensor(buf192, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf192  # reuse
        # Source Nodes: [x_360, x_364, x_se_60, x_se_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf191, buf193, arg364_1, arg365_1, arg100_1, arg101_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg100_1
        del arg101_1
        del arg364_1
        del arg365_1
        # Source Nodes: [x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean]
        buf194 = extern_kernels.convolution(buf193, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg235_1
        del buf193
        buf195 = buf194; del buf194  # reuse
        # Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf195, arg236_1, 896, grid=grid(896), stream=stream0)
        del arg236_1
        # Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf196 = extern_kernels.convolution(buf195, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg237_1
        del buf195
        buf197 = buf191; del buf191  # reuse
        # Source Nodes: [sigmoid_15, x_365, x_366, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf197, buf196, arg238_1, 702464, grid=grid(702464), stream=stream0)
        del arg238_1
        # Source Nodes: [sigmoid_15, x_365, x_366, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf198 = extern_kernels.convolution(buf197, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg239_1
        del buf197
        buf199 = buf187; del buf187  # reuse
        # Source Nodes: [shortcut_16, x_367, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf199, buf198, arg366_1, arg367_1, arg102_1, arg103_1, 702464, grid=grid(702464), stream=stream0)
        del arg102_1
        del arg103_1
        del arg366_1
        del arg367_1
        del buf198
        # Source Nodes: [x_375], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg240_1
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [x_376, x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf201, arg368_1, arg369_1, arg104_1, arg105_1, 702464, grid=grid(702464), stream=stream0)
        del arg104_1
        del arg105_1
        del arg368_1
        del arg369_1
        # Source Nodes: [x_376, x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf202 = extern_kernels.convolution(buf201, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf202, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg241_1
        del buf201
        buf203 = buf202; del buf202  # reuse
        buf204 = reinterpret_tensor(buf196, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf196  # reuse
        buf205 = reinterpret_tensor(buf204, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf204  # reuse
        # Source Nodes: [x_382, x_386, x_se_64, x_se_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf203, buf205, arg370_1, arg371_1, arg106_1, arg107_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg106_1
        del arg107_1
        del arg370_1
        del arg371_1
        # Source Nodes: [x_se_64, x_se_65], Original ATen: [aten.convolution, aten.mean]
        buf206 = extern_kernels.convolution(buf205, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg242_1
        del buf205
        buf207 = buf206; del buf206  # reuse
        # Source Nodes: [x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf207, arg243_1, 896, grid=grid(896), stream=stream0)
        del arg243_1
        # Source Nodes: [x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf208 = extern_kernels.convolution(buf207, arg244_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg244_1
        del buf207
        buf209 = buf203; del buf203  # reuse
        # Source Nodes: [sigmoid_16, x_387, x_388, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf209, buf208, arg245_1, 702464, grid=grid(702464), stream=stream0)
        del arg245_1
        # Source Nodes: [sigmoid_16, x_387, x_388, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf210 = extern_kernels.convolution(buf209, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg246_1
        del buf209
        buf211 = buf199; del buf199  # reuse
        # Source Nodes: [shortcut_17, x_389, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf211, buf210, arg372_1, arg373_1, arg108_1, arg109_1, 702464, grid=grid(702464), stream=stream0)
        del arg108_1
        del arg109_1
        del arg372_1
        del arg373_1
        del buf210
        # Source Nodes: [x_397], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg247_1
        buf213 = buf212; del buf212  # reuse
        # Source Nodes: [x_398, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf213, arg374_1, arg375_1, arg110_1, arg111_1, 702464, grid=grid(702464), stream=stream0)
        del arg110_1
        del arg111_1
        del arg374_1
        del arg375_1
        # Source Nodes: [x_398, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf214 = extern_kernels.convolution(buf213, arg248_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf214, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg248_1
        del buf213
        buf215 = buf214; del buf214  # reuse
        buf216 = reinterpret_tensor(buf208, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf208  # reuse
        buf217 = reinterpret_tensor(buf216, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf216  # reuse
        # Source Nodes: [x_404, x_408, x_se_68, x_se_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17.run(buf215, buf217, arg376_1, arg377_1, arg112_1, arg113_1, 3584, 196, grid=grid(3584), stream=stream0)
        del arg112_1
        del arg113_1
        del arg376_1
        del arg377_1
        # Source Nodes: [x_se_68, x_se_69], Original ATen: [aten.convolution, aten.mean]
        buf218 = extern_kernels.convolution(buf217, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg249_1
        del buf217
        buf219 = buf218; del buf218  # reuse
        # Source Nodes: [x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf219, arg250_1, 896, grid=grid(896), stream=stream0)
        del arg250_1
        # Source Nodes: [x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf220 = extern_kernels.convolution(buf219, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 896, 1, 1), (896, 1, 1, 1))
        del arg251_1
        del buf219
        buf221 = buf215; del buf215  # reuse
        # Source Nodes: [sigmoid_17, x_409, x_410, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_18.run(buf221, buf220, arg252_1, 702464, grid=grid(702464), stream=stream0)
        del arg252_1
        del buf220
        # Source Nodes: [sigmoid_17, x_409, x_410, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf222 = extern_kernels.convolution(buf221, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 896, 14, 14), (175616, 196, 14, 1))
        del arg253_1
        del buf221
        buf223 = buf211; del buf211  # reuse
        # Source Nodes: [shortcut_18, x_411, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf223, buf222, arg378_1, arg379_1, arg114_1, arg115_1, 702464, grid=grid(702464), stream=stream0)
        del arg114_1
        del arg115_1
        del arg378_1
        del arg379_1
        del buf222
        # Source Nodes: [x_420], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 2240, 14, 14), (439040, 196, 14, 1))
        del arg254_1
        buf225 = buf224; del buf224  # reuse
        # Source Nodes: [x_421, x_425, x_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf225, arg380_1, arg381_1, arg116_1, arg117_1, 1756160, grid=grid(1756160), stream=stream0)
        del arg116_1
        del arg117_1
        del arg380_1
        del arg381_1
        # Source Nodes: [x_421, x_425, x_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf226 = extern_kernels.convolution(buf225, arg255_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf226, (4, 2240, 7, 7), (109760, 49, 7, 1))
        del arg255_1
        del buf225
        buf227 = buf226; del buf226  # reuse
        buf228 = empty_strided((4, 2240, 1, 1), (2240, 1, 8960, 8960), device='cuda', dtype=torch.float32)
        buf229 = reinterpret_tensor(buf228, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf228  # reuse
        # Source Nodes: [x_427, x_431, x_se_72, x_se_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24.run(buf227, buf229, arg382_1, arg383_1, arg118_1, arg119_1, 8960, 49, grid=grid(8960), stream=stream0)
        del arg118_1
        del arg119_1
        del arg382_1
        del arg383_1
        # Source Nodes: [x_se_72, x_se_73], Original ATen: [aten.convolution, aten.mean]
        buf230 = extern_kernels.convolution(buf229, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 224, 1, 1), (224, 1, 1, 1))
        del arg256_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Source Nodes: [x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_21.run(buf231, arg257_1, 896, grid=grid(896), stream=stream0)
        del arg257_1
        # Source Nodes: [x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf232 = extern_kernels.convolution(buf231, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 2240, 1, 1), (2240, 1, 1, 1))
        del arg258_1
        del buf231
        buf233 = buf227; del buf227  # reuse
        # Source Nodes: [sigmoid_18, x_432, x_433, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_25.run(buf233, buf232, arg259_1, 439040, grid=grid(439040), stream=stream0)
        del arg259_1
        # Source Nodes: [sigmoid_18, x_432, x_433, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf234 = extern_kernels.convolution(buf233, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 2240, 7, 7), (109760, 49, 7, 1))
        del arg260_1
        del buf233
        # Source Nodes: [x_439], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf223, arg261_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 2240, 7, 7), (109760, 49, 7, 1))
        del arg261_1
        del buf223
        buf236 = buf234; del buf234  # reuse
        buf237 = reinterpret_tensor(buf232, (4, 2240, 1, 1), (2240, 1, 8960, 8960), 0); del buf232  # reuse
        buf238 = reinterpret_tensor(buf237, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf237  # reuse
        # Source Nodes: [x_434, x_440, x_444, x_447, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_26.run(buf236, buf238, arg384_1, arg385_1, arg120_1, arg121_1, buf235, arg386_1, arg387_1, arg122_1, arg123_1, 8960, 49, grid=grid(8960), stream=stream0)
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del arg384_1
        del arg385_1
        del arg386_1
        del arg387_1
        del buf235
        del buf236
        buf239 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg263_1, reinterpret_tensor(buf238, (4, 2240), (2240, 1), 0), reinterpret_tensor(arg262_1, (2240, 1000), (1, 2240), 0), alpha=1, beta=1, out=buf239)
        del arg262_1
        del arg263_1
        return (buf239, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((8, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((224, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((56, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((224, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((56, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((448, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((112, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((896, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2240, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((224, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((2240, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2240, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1000, 2240), (2240, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_regnet', benchmark_compiled_module)
