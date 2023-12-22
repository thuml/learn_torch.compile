
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


# kernel path: /tmp/torchinductor_youkaichao/ig/cigvwrkg75ocxoivxijsq4kzdgneszxghu2zm2ov5makv7ftegxf.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdsqrxb5qhygbazh6d54x2eq2wuhiwhiskmeuzq4uft64adjkro.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_11 => relu_1
# x_7 => add_3, mul_4, mul_5, sub_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11239424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 224
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jnbodth7w7fyuku4kurjzhh57cw3auchcuxfqgm2xvptgmpo2m.py
# Source Nodes: [x_13, x_17, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_13 => add_5, mul_7, mul_8, sub_2
# x_17 => relu_2
# x_se => mean
triton_red_fused__native_batch_norm_legit_no_training_mean_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_relu_2', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 224
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr0 + (r2 + (3136*x3)), tmp15, rmask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = 3136.0
    tmp20 = tmp17 / tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehalze2gttbyfb3gsibunlzogcp3nlldtl6tij2j22y7vgsmvxl.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_3
# x_se_2 => relu_3
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/7w/c7whz4ad5mgney2yby7nkhqpdzrpaqqus4cjryz4exggvxgddyzf.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_4
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': ['in_out_ptr0']},
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr2delctes2rcgi2ai7kv5tqwihlvlwj673vhxmeybeeda2lpsq.py
# Source Nodes: [sigmoid, x_18], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid => sigmoid
# x_18 => mul_9
triton_poi_fused_mul_sigmoid_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czi4v6paklnl3p3t7uv35fxwxgh3wehjnacytpmk3iqarf6vezpo.py
# Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_1 => relu_4
# x_20 => add_7, mul_11, mul_12, sub_3
# x_26 => add_9, mul_14, mul_15, sub_4
# x_30 => add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/tv/ctveal2uhxfp2nd5gs2c65i2fqy5jj6v34jw4nnujzgq3ravhbsj.py
# Source Nodes: [x_34, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_34 => add_12, mul_17, mul_18, sub_5
# x_38 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2oofeh2eouq4rd5mksixebojmmrxa5lew5m7slcolilxd4n5px.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_9
# x_se_6 => relu_7
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5pby3lzuvytc2v7bam7jc2zdsjnx4xpiziitjvcnsob6q4ibnc.py
# Source Nodes: [shortcut_2, x_47, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_2 => relu_8
# x_47 => add_16, mul_24, mul_25, sub_7
# x_52 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e4b4hdztwv52n5ixcl5lkkdqb5lnm6xtwjzltkuupfhsqmxemf.py
# Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_57 => add_19, mul_27, mul_28, sub_8
# x_61 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnkdfiejnukbpxhynifdgapflmiprwj5chvznubjnlyu2d35itr.py
# Source Nodes: [x_63, x_67, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_63 => add_21, mul_30, mul_31, sub_9
# x_67 => relu_10
# x_se_8 => mean_2
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckuxdcg6gbugundzoaxrarybvmntca6fxsrunp2zoaxtz34tudpv.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_15
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvk5fz7oen6up6f6nyvpr6kkafcpfjme5ztijmf22wpnukkarh5t.py
# Source Nodes: [sigmoid_2, x_68], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_2 => sigmoid_2
# x_68 => mul_32
triton_poi_fused_mul_sigmoid_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4xb7yp5ctyg2urpvbhbqbvdux7ottu45o6veehqpmy5sp3zqi4.py
# Source Nodes: [shortcut_3, x_70, x_76, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_12
# x_70 => add_23, mul_34, mul_35, sub_10
# x_76 => add_25, mul_37, mul_38, sub_11
# x_80 => add_26
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyojiyxeasewzyptkjj2ocqv43uq75rch5mzpp53jzgduisuc4u.py
# Source Nodes: [x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_84 => add_28, mul_40, mul_41, sub_12
# x_88 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3clremwzkhxuyyphcdjdo5rlosibqnhcqitx5qaj5afffhgxjeg.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_20
# x_se_14 => relu_15
triton_poi_fused_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/yg/cygb3gqiaxby735fd5lowweno2ewhdks4lty5w3qbpk6kerjybgu.py
# Source Nodes: [shortcut_4, x_102, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_4 => relu_16
# x_102 => add_33
# x_97 => add_32, mul_47, mul_48, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74d6xaz3isd7xo7jaqp2bszoygll4i47jxsdwzzcyjxy2szryis.py
# Source Nodes: [x_173, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_173 => add_56, mul_80, mul_81, sub_24
# x_177 => relu_29
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chiq5ipzwp3seqdmahr4wbdvqpchv6akto2ich4q5f26jqb6j4h7.py
# Source Nodes: [x_179, x_183, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_179 => add_58, mul_83, mul_84, sub_25
# x_183 => relu_30
# x_se_28 => mean_7
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
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
    tmp20 = 196.0
    tmp21 = tmp19 / tmp20
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csokrxc6ja6oa3no3mtylzsm3tkwwpk3gzrrvwzbalbjxbrywu4h.py
# Source Nodes: [x_se_31], Original ATen: [aten.convolution]
# x_se_31 => convolution_41
triton_poi_fused_convolution_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 896
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmvae644wjgexhjt4ijvzzohesgr5m3wr3do4dum3wrn56lmwom.py
# Source Nodes: [sigmoid_7, x_184], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_7 => sigmoid_7
# x_184 => mul_85
triton_poi_fused_mul_sigmoid_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjb3iq6aa3ab36eqeaxsgx43kabwjn3rqoqththvvejo72xjxsxo.py
# Source Nodes: [shortcut_8, x_186, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_8 => relu_32
# x_186 => add_60, mul_87, mul_88, sub_26
# x_192 => add_62, mul_90, mul_91, sub_27
# x_196 => add_63
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/nt/cntswa5bhpjtzb2hm5tjrc7pr3y7tbjxsnprdc7g3aa5nt2olr2y.py
# Source Nodes: [x_200, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_200 => add_65, mul_93, mul_94, sub_28
# x_204 => relu_33
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahf6mnr2crj2kcnskreps6reb32xzoierw7c4ppjbeblrnzpxnl.py
# Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.relu]
# x_se_33 => convolution_46
# x_se_34 => relu_35
triton_poi_fused_convolution_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/sl/cslevjsgzqjthonfgz5nq2grxtbggm7b3lpcxsl5q24dbgqqizp4.py
# Source Nodes: [shortcut_9, x_213, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_9 => relu_36
# x_213 => add_69, mul_100, mul_101, sub_30
# x_218 => add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5f/c5ffzc3jxp7k7h45ijp3palu5wtsuysu2kxxzqzmmoeudbgey7se.py
# Source Nodes: [x_421, x_425], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_421 => add_135, mul_193, mul_194, sub_58
# x_425 => relu_73
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1756160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czs5edxi7e7ly5ocisnn3fwg2z57y3trrgnwfrx3wjqtfebn6kdo.py
# Source Nodes: [x_427, x_431, x_se_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_427 => add_137, mul_196, mul_197, sub_59
# x_431 => relu_74
# x_se_72 => mean_18
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfeezynsya2wjm46ykdwjybdfnpj7q3gzls4rbmbfnn6yau7v6dz.py
# Source Nodes: [x_se_75], Original ATen: [aten.convolution]
# x_se_75 => convolution_97
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dngtewt74nz6hqx6zfwf3illhy7wxf2szf5feep6gmbwqbuei6.py
# Source Nodes: [sigmoid_18, x_432], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_18 => sigmoid_18
# x_432 => mul_198
triton_poi_fused_mul_sigmoid_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csv6ukn55bqcppiinkh2jrxr6rvcx2ckdo7z3ycig2r6opn2jq32.py
# Source Nodes: [x_434, x_440, x_444, x_447, x_450, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# x_434 => add_139, mul_200, mul_201, sub_60
# x_440 => add_141, mul_203, mul_204, sub_61
# x_444 => add_142
# x_447 => relu_76
# x_450 => mean_19
# x_452 => view
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*i1', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
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
    tmp30 = 0.0
    tmp31 = tmp29 <= tmp30
    tmp32 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = 49.0
    tmp37 = tmp35 / tmp36
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp31, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp37, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (224, ), (1, ))
    assert_size_stride(primals_4, (224, ), (1, ))
    assert_size_stride(primals_5, (224, ), (1, ))
    assert_size_stride(primals_6, (224, ), (1, ))
    assert_size_stride(primals_7, (224, ), (1, ))
    assert_size_stride(primals_8, (224, ), (1, ))
    assert_size_stride(primals_9, (224, ), (1, ))
    assert_size_stride(primals_10, (224, ), (1, ))
    assert_size_stride(primals_11, (224, ), (1, ))
    assert_size_stride(primals_12, (224, ), (1, ))
    assert_size_stride(primals_13, (224, ), (1, ))
    assert_size_stride(primals_14, (224, ), (1, ))
    assert_size_stride(primals_15, (224, ), (1, ))
    assert_size_stride(primals_16, (224, ), (1, ))
    assert_size_stride(primals_17, (448, ), (1, ))
    assert_size_stride(primals_18, (448, ), (1, ))
    assert_size_stride(primals_19, (448, ), (1, ))
    assert_size_stride(primals_20, (448, ), (1, ))
    assert_size_stride(primals_21, (448, ), (1, ))
    assert_size_stride(primals_22, (448, ), (1, ))
    assert_size_stride(primals_23, (448, ), (1, ))
    assert_size_stride(primals_24, (448, ), (1, ))
    assert_size_stride(primals_25, (448, ), (1, ))
    assert_size_stride(primals_26, (448, ), (1, ))
    assert_size_stride(primals_27, (448, ), (1, ))
    assert_size_stride(primals_28, (448, ), (1, ))
    assert_size_stride(primals_29, (448, ), (1, ))
    assert_size_stride(primals_30, (448, ), (1, ))
    assert_size_stride(primals_31, (448, ), (1, ))
    assert_size_stride(primals_32, (448, ), (1, ))
    assert_size_stride(primals_33, (448, ), (1, ))
    assert_size_stride(primals_34, (448, ), (1, ))
    assert_size_stride(primals_35, (448, ), (1, ))
    assert_size_stride(primals_36, (448, ), (1, ))
    assert_size_stride(primals_37, (448, ), (1, ))
    assert_size_stride(primals_38, (448, ), (1, ))
    assert_size_stride(primals_39, (448, ), (1, ))
    assert_size_stride(primals_40, (448, ), (1, ))
    assert_size_stride(primals_41, (448, ), (1, ))
    assert_size_stride(primals_42, (448, ), (1, ))
    assert_size_stride(primals_43, (448, ), (1, ))
    assert_size_stride(primals_44, (448, ), (1, ))
    assert_size_stride(primals_45, (448, ), (1, ))
    assert_size_stride(primals_46, (448, ), (1, ))
    assert_size_stride(primals_47, (448, ), (1, ))
    assert_size_stride(primals_48, (448, ), (1, ))
    assert_size_stride(primals_49, (896, ), (1, ))
    assert_size_stride(primals_50, (896, ), (1, ))
    assert_size_stride(primals_51, (896, ), (1, ))
    assert_size_stride(primals_52, (896, ), (1, ))
    assert_size_stride(primals_53, (896, ), (1, ))
    assert_size_stride(primals_54, (896, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_56, (896, ), (1, ))
    assert_size_stride(primals_57, (896, ), (1, ))
    assert_size_stride(primals_58, (896, ), (1, ))
    assert_size_stride(primals_59, (896, ), (1, ))
    assert_size_stride(primals_60, (896, ), (1, ))
    assert_size_stride(primals_61, (896, ), (1, ))
    assert_size_stride(primals_62, (896, ), (1, ))
    assert_size_stride(primals_63, (896, ), (1, ))
    assert_size_stride(primals_64, (896, ), (1, ))
    assert_size_stride(primals_65, (896, ), (1, ))
    assert_size_stride(primals_66, (896, ), (1, ))
    assert_size_stride(primals_67, (896, ), (1, ))
    assert_size_stride(primals_68, (896, ), (1, ))
    assert_size_stride(primals_69, (896, ), (1, ))
    assert_size_stride(primals_70, (896, ), (1, ))
    assert_size_stride(primals_71, (896, ), (1, ))
    assert_size_stride(primals_72, (896, ), (1, ))
    assert_size_stride(primals_73, (896, ), (1, ))
    assert_size_stride(primals_74, (896, ), (1, ))
    assert_size_stride(primals_75, (896, ), (1, ))
    assert_size_stride(primals_76, (896, ), (1, ))
    assert_size_stride(primals_77, (896, ), (1, ))
    assert_size_stride(primals_78, (896, ), (1, ))
    assert_size_stride(primals_79, (896, ), (1, ))
    assert_size_stride(primals_80, (896, ), (1, ))
    assert_size_stride(primals_81, (896, ), (1, ))
    assert_size_stride(primals_82, (896, ), (1, ))
    assert_size_stride(primals_83, (896, ), (1, ))
    assert_size_stride(primals_84, (896, ), (1, ))
    assert_size_stride(primals_85, (896, ), (1, ))
    assert_size_stride(primals_86, (896, ), (1, ))
    assert_size_stride(primals_87, (896, ), (1, ))
    assert_size_stride(primals_88, (896, ), (1, ))
    assert_size_stride(primals_89, (896, ), (1, ))
    assert_size_stride(primals_90, (896, ), (1, ))
    assert_size_stride(primals_91, (896, ), (1, ))
    assert_size_stride(primals_92, (896, ), (1, ))
    assert_size_stride(primals_93, (896, ), (1, ))
    assert_size_stride(primals_94, (896, ), (1, ))
    assert_size_stride(primals_95, (896, ), (1, ))
    assert_size_stride(primals_96, (896, ), (1, ))
    assert_size_stride(primals_97, (896, ), (1, ))
    assert_size_stride(primals_98, (896, ), (1, ))
    assert_size_stride(primals_99, (896, ), (1, ))
    assert_size_stride(primals_100, (896, ), (1, ))
    assert_size_stride(primals_101, (896, ), (1, ))
    assert_size_stride(primals_102, (896, ), (1, ))
    assert_size_stride(primals_103, (896, ), (1, ))
    assert_size_stride(primals_104, (896, ), (1, ))
    assert_size_stride(primals_105, (896, ), (1, ))
    assert_size_stride(primals_106, (896, ), (1, ))
    assert_size_stride(primals_107, (896, ), (1, ))
    assert_size_stride(primals_108, (896, ), (1, ))
    assert_size_stride(primals_109, (896, ), (1, ))
    assert_size_stride(primals_110, (896, ), (1, ))
    assert_size_stride(primals_111, (896, ), (1, ))
    assert_size_stride(primals_112, (896, ), (1, ))
    assert_size_stride(primals_113, (896, ), (1, ))
    assert_size_stride(primals_114, (896, ), (1, ))
    assert_size_stride(primals_115, (896, ), (1, ))
    assert_size_stride(primals_116, (896, ), (1, ))
    assert_size_stride(primals_117, (2240, ), (1, ))
    assert_size_stride(primals_118, (2240, ), (1, ))
    assert_size_stride(primals_119, (2240, ), (1, ))
    assert_size_stride(primals_120, (2240, ), (1, ))
    assert_size_stride(primals_121, (2240, ), (1, ))
    assert_size_stride(primals_122, (2240, ), (1, ))
    assert_size_stride(primals_123, (2240, ), (1, ))
    assert_size_stride(primals_124, (2240, ), (1, ))
    assert_size_stride(primals_125, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_126, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_127, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_128, (8, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_129, (8, ), (1, ))
    assert_size_stride(primals_130, (224, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_131, (224, ), (1, ))
    assert_size_stride(primals_132, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_133, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_135, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_136, (56, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_137, (56, ), (1, ))
    assert_size_stride(primals_138, (224, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_139, (224, ), (1, ))
    assert_size_stride(primals_140, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_141, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_142, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_143, (56, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_144, (56, ), (1, ))
    assert_size_stride(primals_145, (448, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_146, (448, ), (1, ))
    assert_size_stride(primals_147, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_148, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_149, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_150, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_151, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_152, (112, ), (1, ))
    assert_size_stride(primals_153, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_154, (448, ), (1, ))
    assert_size_stride(primals_155, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_156, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_157, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_158, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_159, (112, ), (1, ))
    assert_size_stride(primals_160, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_161, (448, ), (1, ))
    assert_size_stride(primals_162, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_163, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_164, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_165, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_166, (112, ), (1, ))
    assert_size_stride(primals_167, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_168, (448, ), (1, ))
    assert_size_stride(primals_169, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_170, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_171, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_172, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_173, (112, ), (1, ))
    assert_size_stride(primals_174, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_175, (448, ), (1, ))
    assert_size_stride(primals_176, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_177, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_178, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_179, (112, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_180, (112, ), (1, ))
    assert_size_stride(primals_181, (896, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_182, (896, ), (1, ))
    assert_size_stride(primals_183, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_184, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_185, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_186, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_187, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_188, (224, ), (1, ))
    assert_size_stride(primals_189, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_190, (896, ), (1, ))
    assert_size_stride(primals_191, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_192, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_193, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_194, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_195, (224, ), (1, ))
    assert_size_stride(primals_196, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_197, (896, ), (1, ))
    assert_size_stride(primals_198, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_199, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_200, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_201, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_202, (224, ), (1, ))
    assert_size_stride(primals_203, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_204, (896, ), (1, ))
    assert_size_stride(primals_205, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_206, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_207, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_208, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_209, (224, ), (1, ))
    assert_size_stride(primals_210, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_211, (896, ), (1, ))
    assert_size_stride(primals_212, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_213, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_214, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_215, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_216, (224, ), (1, ))
    assert_size_stride(primals_217, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_218, (896, ), (1, ))
    assert_size_stride(primals_219, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_220, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_221, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_222, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_223, (224, ), (1, ))
    assert_size_stride(primals_224, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_225, (896, ), (1, ))
    assert_size_stride(primals_226, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_227, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_228, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_229, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_230, (224, ), (1, ))
    assert_size_stride(primals_231, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_232, (896, ), (1, ))
    assert_size_stride(primals_233, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_234, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_235, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_236, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_237, (224, ), (1, ))
    assert_size_stride(primals_238, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_239, (896, ), (1, ))
    assert_size_stride(primals_240, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_241, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_242, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_243, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_244, (224, ), (1, ))
    assert_size_stride(primals_245, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_246, (896, ), (1, ))
    assert_size_stride(primals_247, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_248, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_249, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_250, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_251, (224, ), (1, ))
    assert_size_stride(primals_252, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_253, (896, ), (1, ))
    assert_size_stride(primals_254, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_255, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_256, (2240, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_257, (224, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_258, (224, ), (1, ))
    assert_size_stride(primals_259, (2240, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_260, (2240, ), (1, ))
    assert_size_stride(primals_261, (2240, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_262, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_263, (1000, 2240), (2240, 1))
    assert_size_stride(primals_264, (1000, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (224, ), (1, ))
    assert_size_stride(primals_268, (224, ), (1, ))
    assert_size_stride(primals_269, (224, ), (1, ))
    assert_size_stride(primals_270, (224, ), (1, ))
    assert_size_stride(primals_271, (224, ), (1, ))
    assert_size_stride(primals_272, (224, ), (1, ))
    assert_size_stride(primals_273, (224, ), (1, ))
    assert_size_stride(primals_274, (224, ), (1, ))
    assert_size_stride(primals_275, (224, ), (1, ))
    assert_size_stride(primals_276, (224, ), (1, ))
    assert_size_stride(primals_277, (224, ), (1, ))
    assert_size_stride(primals_278, (224, ), (1, ))
    assert_size_stride(primals_279, (224, ), (1, ))
    assert_size_stride(primals_280, (224, ), (1, ))
    assert_size_stride(primals_281, (448, ), (1, ))
    assert_size_stride(primals_282, (448, ), (1, ))
    assert_size_stride(primals_283, (448, ), (1, ))
    assert_size_stride(primals_284, (448, ), (1, ))
    assert_size_stride(primals_285, (448, ), (1, ))
    assert_size_stride(primals_286, (448, ), (1, ))
    assert_size_stride(primals_287, (448, ), (1, ))
    assert_size_stride(primals_288, (448, ), (1, ))
    assert_size_stride(primals_289, (448, ), (1, ))
    assert_size_stride(primals_290, (448, ), (1, ))
    assert_size_stride(primals_291, (448, ), (1, ))
    assert_size_stride(primals_292, (448, ), (1, ))
    assert_size_stride(primals_293, (448, ), (1, ))
    assert_size_stride(primals_294, (448, ), (1, ))
    assert_size_stride(primals_295, (448, ), (1, ))
    assert_size_stride(primals_296, (448, ), (1, ))
    assert_size_stride(primals_297, (448, ), (1, ))
    assert_size_stride(primals_298, (448, ), (1, ))
    assert_size_stride(primals_299, (448, ), (1, ))
    assert_size_stride(primals_300, (448, ), (1, ))
    assert_size_stride(primals_301, (448, ), (1, ))
    assert_size_stride(primals_302, (448, ), (1, ))
    assert_size_stride(primals_303, (448, ), (1, ))
    assert_size_stride(primals_304, (448, ), (1, ))
    assert_size_stride(primals_305, (448, ), (1, ))
    assert_size_stride(primals_306, (448, ), (1, ))
    assert_size_stride(primals_307, (448, ), (1, ))
    assert_size_stride(primals_308, (448, ), (1, ))
    assert_size_stride(primals_309, (448, ), (1, ))
    assert_size_stride(primals_310, (448, ), (1, ))
    assert_size_stride(primals_311, (448, ), (1, ))
    assert_size_stride(primals_312, (448, ), (1, ))
    assert_size_stride(primals_313, (896, ), (1, ))
    assert_size_stride(primals_314, (896, ), (1, ))
    assert_size_stride(primals_315, (896, ), (1, ))
    assert_size_stride(primals_316, (896, ), (1, ))
    assert_size_stride(primals_317, (896, ), (1, ))
    assert_size_stride(primals_318, (896, ), (1, ))
    assert_size_stride(primals_319, (896, ), (1, ))
    assert_size_stride(primals_320, (896, ), (1, ))
    assert_size_stride(primals_321, (896, ), (1, ))
    assert_size_stride(primals_322, (896, ), (1, ))
    assert_size_stride(primals_323, (896, ), (1, ))
    assert_size_stride(primals_324, (896, ), (1, ))
    assert_size_stride(primals_325, (896, ), (1, ))
    assert_size_stride(primals_326, (896, ), (1, ))
    assert_size_stride(primals_327, (896, ), (1, ))
    assert_size_stride(primals_328, (896, ), (1, ))
    assert_size_stride(primals_329, (896, ), (1, ))
    assert_size_stride(primals_330, (896, ), (1, ))
    assert_size_stride(primals_331, (896, ), (1, ))
    assert_size_stride(primals_332, (896, ), (1, ))
    assert_size_stride(primals_333, (896, ), (1, ))
    assert_size_stride(primals_334, (896, ), (1, ))
    assert_size_stride(primals_335, (896, ), (1, ))
    assert_size_stride(primals_336, (896, ), (1, ))
    assert_size_stride(primals_337, (896, ), (1, ))
    assert_size_stride(primals_338, (896, ), (1, ))
    assert_size_stride(primals_339, (896, ), (1, ))
    assert_size_stride(primals_340, (896, ), (1, ))
    assert_size_stride(primals_341, (896, ), (1, ))
    assert_size_stride(primals_342, (896, ), (1, ))
    assert_size_stride(primals_343, (896, ), (1, ))
    assert_size_stride(primals_344, (896, ), (1, ))
    assert_size_stride(primals_345, (896, ), (1, ))
    assert_size_stride(primals_346, (896, ), (1, ))
    assert_size_stride(primals_347, (896, ), (1, ))
    assert_size_stride(primals_348, (896, ), (1, ))
    assert_size_stride(primals_349, (896, ), (1, ))
    assert_size_stride(primals_350, (896, ), (1, ))
    assert_size_stride(primals_351, (896, ), (1, ))
    assert_size_stride(primals_352, (896, ), (1, ))
    assert_size_stride(primals_353, (896, ), (1, ))
    assert_size_stride(primals_354, (896, ), (1, ))
    assert_size_stride(primals_355, (896, ), (1, ))
    assert_size_stride(primals_356, (896, ), (1, ))
    assert_size_stride(primals_357, (896, ), (1, ))
    assert_size_stride(primals_358, (896, ), (1, ))
    assert_size_stride(primals_359, (896, ), (1, ))
    assert_size_stride(primals_360, (896, ), (1, ))
    assert_size_stride(primals_361, (896, ), (1, ))
    assert_size_stride(primals_362, (896, ), (1, ))
    assert_size_stride(primals_363, (896, ), (1, ))
    assert_size_stride(primals_364, (896, ), (1, ))
    assert_size_stride(primals_365, (896, ), (1, ))
    assert_size_stride(primals_366, (896, ), (1, ))
    assert_size_stride(primals_367, (896, ), (1, ))
    assert_size_stride(primals_368, (896, ), (1, ))
    assert_size_stride(primals_369, (896, ), (1, ))
    assert_size_stride(primals_370, (896, ), (1, ))
    assert_size_stride(primals_371, (896, ), (1, ))
    assert_size_stride(primals_372, (896, ), (1, ))
    assert_size_stride(primals_373, (896, ), (1, ))
    assert_size_stride(primals_374, (896, ), (1, ))
    assert_size_stride(primals_375, (896, ), (1, ))
    assert_size_stride(primals_376, (896, ), (1, ))
    assert_size_stride(primals_377, (896, ), (1, ))
    assert_size_stride(primals_378, (896, ), (1, ))
    assert_size_stride(primals_379, (896, ), (1, ))
    assert_size_stride(primals_380, (896, ), (1, ))
    assert_size_stride(primals_381, (2240, ), (1, ))
    assert_size_stride(primals_382, (2240, ), (1, ))
    assert_size_stride(primals_383, (2240, ), (1, ))
    assert_size_stride(primals_384, (2240, ), (1, ))
    assert_size_stride(primals_385, (2240, ), (1, ))
    assert_size_stride(primals_386, (2240, ), (1, ))
    assert_size_stride(primals_387, (2240, ), (1, ))
    assert_size_stride(primals_388, (2240, ), (1, ))
    assert_size_stride(primals_389, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_389, primals_125, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf1 = empty((4, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_265, primals_266, primals_1, primals_2, buf1, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 224, 112, 112), (2809856, 12544, 112, 1))
        buf3 = empty((4, 224, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf2, primals_267, primals_268, primals_3, primals_4, buf3, 11239424, grid=grid(11239424), stream=stream0)
        del primals_4
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_127, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf4, (4, 224, 56, 56), (702464, 3136, 56, 1))
        buf5 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf6, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf6  # reuse
        # Source Nodes: [x_13, x_17, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_relu_2.run(buf7, buf4, primals_269, primals_270, primals_5, primals_6, buf5, 896, 3136, grid=grid(896), stream=stream0)
        del primals_6
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 8, 1, 1), (8, 1, 1, 1))
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_3.run(buf9, primals_129, 32, grid=grid(32), stream=stream0)
        del primals_129
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 224, 1, 1), (224, 1, 1, 1))
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf11, primals_131, 896, grid=grid(896), stream=stream0)
        del primals_131
        buf12 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_18], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_5.run(buf5, buf11, buf12, 2809856, grid=grid(2809856), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 224, 56, 56), (702464, 3136, 56, 1))
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf1, primals_133, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 224, 56, 56), (702464, 3136, 56, 1))
        buf15 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf16, buf13, primals_271, primals_272, primals_7, primals_8, buf14, primals_273, primals_274, primals_9, primals_10, 2809856, grid=grid(2809856), stream=stream0)
        del primals_10
        del primals_8
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 224, 56, 56), (702464, 3136, 56, 1))
        buf18 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf17, primals_275, primals_276, primals_11, primals_12, buf18, 2809856, grid=grid(2809856), stream=stream0)
        del primals_12
        # Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf19, (4, 224, 56, 56), (702464, 3136, 56, 1))
        buf20 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf22 = reinterpret_tensor(buf21, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf21  # reuse
        # Source Nodes: [x_40, x_44, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_relu_2.run(buf22, buf19, primals_277, primals_278, primals_13, primals_14, buf20, 896, 3136, grid=grid(896), stream=stream0)
        del primals_14
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 56, 1, 1), (56, 1, 1, 1))
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf24, primals_137, 224, grid=grid(224), stream=stream0)
        del primals_137
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 224, 1, 1), (224, 1, 1, 1))
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf26, primals_139, 896, grid=grid(896), stream=stream0)
        del primals_139
        buf27 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_45], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_5.run(buf20, buf26, buf27, 2809856, grid=grid(2809856), stream=stream0)
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 224, 56, 56), (702464, 3136, 56, 1))
        buf29 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_2, x_47, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf28, primals_279, primals_280, primals_15, primals_16, buf16, buf29, 2809856, grid=grid(2809856), stream=stream0)
        del primals_16
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 448, 56, 56), (1404928, 3136, 56, 1))
        buf31 = empty((4, 448, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf30, primals_281, primals_282, primals_17, primals_18, buf31, 5619712, grid=grid(5619712), stream=stream0)
        del primals_18
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_142, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf32, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf33 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf35 = reinterpret_tensor(buf34, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf34  # reuse
        # Source Nodes: [x_63, x_67, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11.run(buf35, buf32, primals_283, primals_284, primals_19, primals_20, buf33, 1792, 784, grid=grid(1792), stream=stream0)
        del primals_20
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 56, 1, 1), (56, 1, 1, 1))
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf37, primals_144, 224, grid=grid(224), stream=stream0)
        del primals_144
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 448, 1, 1), (448, 1, 1, 1))
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf39, primals_146, 1792, grid=grid(1792), stream=stream0)
        del primals_146
        buf40 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_68], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_13.run(buf33, buf39, buf40, 1404928, grid=grid(1404928), stream=stream0)
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 448, 28, 28), (351232, 784, 28, 1))
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf29, primals_148, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf43 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf44 = buf43; del buf43  # reuse
        # Source Nodes: [shortcut_3, x_70, x_76, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf44, buf41, primals_285, primals_286, primals_21, primals_22, buf42, primals_287, primals_288, primals_23, primals_24, 1404928, grid=grid(1404928), stream=stream0)
        del primals_22
        del primals_24
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf46 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf45, primals_289, primals_290, primals_25, primals_26, buf46, 1404928, grid=grid(1404928), stream=stream0)
        del primals_26
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf47, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf48 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf50 = reinterpret_tensor(buf49, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf49  # reuse
        # Source Nodes: [x_90, x_94, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11.run(buf50, buf47, primals_291, primals_292, primals_27, primals_28, buf48, 1792, 784, grid=grid(1792), stream=stream0)
        del primals_28
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 112, 1, 1), (112, 1, 1, 1))
        buf52 = buf51; del buf51  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf52, primals_152, 448, grid=grid(448), stream=stream0)
        del primals_152
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 448, 1, 1), (448, 1, 1, 1))
        buf54 = buf53; del buf53  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf54, primals_154, 1792, grid=grid(1792), stream=stream0)
        del primals_154
        buf55 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_95], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_13.run(buf48, buf54, buf55, 1404928, grid=grid(1404928), stream=stream0)
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf57 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_4, x_102, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf56, primals_293, primals_294, primals_29, primals_30, buf44, buf57, 1404928, grid=grid(1404928), stream=stream0)
        del primals_30
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf59 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf58, primals_295, primals_296, primals_31, primals_32, buf59, 1404928, grid=grid(1404928), stream=stream0)
        del primals_32
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf60, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf61 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf63 = reinterpret_tensor(buf62, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf62  # reuse
        # Source Nodes: [x_112, x_116, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11.run(buf63, buf60, primals_297, primals_298, primals_33, primals_34, buf61, 1792, 784, grid=grid(1792), stream=stream0)
        del primals_34
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 112, 1, 1), (112, 1, 1, 1))
        buf65 = buf64; del buf64  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf65, primals_159, 448, grid=grid(448), stream=stream0)
        del primals_159
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 448, 1, 1), (448, 1, 1, 1))
        buf67 = buf66; del buf66  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf67, primals_161, 1792, grid=grid(1792), stream=stream0)
        del primals_161
        buf68 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_117], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_13.run(buf61, buf67, buf68, 1404928, grid=grid(1404928), stream=stream0)
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf70 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_119, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf69, primals_299, primals_300, primals_35, primals_36, buf57, buf70, 1404928, grid=grid(1404928), stream=stream0)
        del primals_36
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf72 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf71, primals_301, primals_302, primals_37, primals_38, buf72, 1404928, grid=grid(1404928), stream=stream0)
        del primals_38
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf73, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf74 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf76 = reinterpret_tensor(buf75, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf75  # reuse
        # Source Nodes: [x_134, x_138, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11.run(buf76, buf73, primals_303, primals_304, primals_39, primals_40, buf74, 1792, 784, grid=grid(1792), stream=stream0)
        del primals_40
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 112, 1, 1), (112, 1, 1, 1))
        buf78 = buf77; del buf77  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf78, primals_166, 448, grid=grid(448), stream=stream0)
        del primals_166
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 448, 1, 1), (448, 1, 1, 1))
        buf80 = buf79; del buf79  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf80, primals_168, 1792, grid=grid(1792), stream=stream0)
        del primals_168
        buf81 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_139], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_13.run(buf74, buf80, buf81, 1404928, grid=grid(1404928), stream=stream0)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf83 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_141, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf82, primals_305, primals_306, primals_41, primals_42, buf70, buf83, 1404928, grid=grid(1404928), stream=stream0)
        del primals_42
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf85 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf84, primals_307, primals_308, primals_43, primals_44, buf85, 1404928, grid=grid(1404928), stream=stream0)
        del primals_44
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf86, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf87 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf89 = reinterpret_tensor(buf88, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf88  # reuse
        # Source Nodes: [x_156, x_160, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_11.run(buf89, buf86, primals_309, primals_310, primals_45, primals_46, buf87, 1792, 784, grid=grid(1792), stream=stream0)
        del primals_46
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 112, 1, 1), (112, 1, 1, 1))
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf91, primals_173, 448, grid=grid(448), stream=stream0)
        del primals_173
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 448, 1, 1), (448, 1, 1, 1))
        buf93 = buf92; del buf92  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf93, primals_175, 1792, grid=grid(1792), stream=stream0)
        del primals_175
        buf94 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_161], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_13.run(buf87, buf93, buf94, 1404928, grid=grid(1404928), stream=stream0)
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 448, 28, 28), (351232, 784, 28, 1))
        buf96 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_163, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf95, primals_311, primals_312, primals_47, primals_48, buf83, buf96, 1404928, grid=grid(1404928), stream=stream0)
        del primals_48
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 896, 28, 28), (702464, 784, 28, 1))
        buf98 = empty((4, 896, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf97, primals_313, primals_314, primals_49, primals_50, buf98, 2809856, grid=grid(2809856), stream=stream0)
        del primals_50
        # Source Nodes: [x_178], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_178, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf99, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf100 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf101 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf102 = reinterpret_tensor(buf101, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf101  # reuse
        # Source Nodes: [x_179, x_183, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf102, buf99, primals_315, primals_316, primals_51, primals_52, buf100, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_52
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 112, 1, 1), (112, 1, 1, 1))
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf104, primals_180, 448, grid=grid(448), stream=stream0)
        del primals_180
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 896, 1, 1), (896, 1, 1, 1))
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf106, primals_182, 3584, grid=grid(3584), stream=stream0)
        del primals_182
        buf107 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_184], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf100, buf106, buf107, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 896, 14, 14), (175616, 196, 14, 1))
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf96, primals_184, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf110 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [shortcut_8, x_186, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf111, buf108, primals_317, primals_318, primals_53, primals_54, buf109, primals_319, primals_320, primals_55, primals_56, 702464, grid=grid(702464), stream=stream0)
        del primals_54
        del primals_56
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf113 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf112, primals_321, primals_322, primals_57, primals_58, buf113, 702464, grid=grid(702464), stream=stream0)
        del primals_58
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf114, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf115 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf116 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf117 = reinterpret_tensor(buf116, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf116  # reuse
        # Source Nodes: [x_206, x_210, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf117, buf114, primals_323, primals_324, primals_59, primals_60, buf115, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_60
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 224, 1, 1), (224, 1, 1, 1))
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf119, primals_188, 896, grid=grid(896), stream=stream0)
        del primals_188
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 896, 1, 1), (896, 1, 1, 1))
        buf121 = buf120; del buf120  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf121, primals_190, 3584, grid=grid(3584), stream=stream0)
        del primals_190
        buf122 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8, x_211], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf115, buf121, buf122, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf124 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_213, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf123, primals_325, primals_326, primals_61, primals_62, buf111, buf124, 702464, grid=grid(702464), stream=stream0)
        del primals_62
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf126 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf125, primals_327, primals_328, primals_63, primals_64, buf126, 702464, grid=grid(702464), stream=stream0)
        del primals_64
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf127, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf128 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf129 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf130 = reinterpret_tensor(buf129, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf129  # reuse
        # Source Nodes: [x_228, x_232, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf130, buf127, primals_329, primals_330, primals_65, primals_66, buf128, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_66
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 224, 1, 1), (224, 1, 1, 1))
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf132, primals_195, 896, grid=grid(896), stream=stream0)
        del primals_195
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 896, 1, 1), (896, 1, 1, 1))
        buf134 = buf133; del buf133  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf134, primals_197, 3584, grid=grid(3584), stream=stream0)
        del primals_197
        buf135 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9, x_233], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf128, buf134, buf135, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf137 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10, x_235, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf136, primals_331, primals_332, primals_67, primals_68, buf124, buf137, 702464, grid=grid(702464), stream=stream0)
        del primals_68
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf139 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf138, primals_333, primals_334, primals_69, primals_70, buf139, 702464, grid=grid(702464), stream=stream0)
        del primals_70
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf140, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf141 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf143 = reinterpret_tensor(buf142, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf142  # reuse
        # Source Nodes: [x_250, x_254, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf143, buf140, primals_335, primals_336, primals_71, primals_72, buf141, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_72
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 224, 1, 1), (224, 1, 1, 1))
        buf145 = buf144; del buf144  # reuse
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf145, primals_202, 896, grid=grid(896), stream=stream0)
        del primals_202
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 896, 1, 1), (896, 1, 1, 1))
        buf147 = buf146; del buf146  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf147, primals_204, 3584, grid=grid(3584), stream=stream0)
        del primals_204
        buf148 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10, x_255], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf141, buf147, buf148, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf150 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_11, x_257, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf149, primals_337, primals_338, primals_73, primals_74, buf137, buf150, 702464, grid=grid(702464), stream=stream0)
        del primals_74
        # Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf152 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf151, primals_339, primals_340, primals_75, primals_76, buf152, 702464, grid=grid(702464), stream=stream0)
        del primals_76
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf153, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf154 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf156 = reinterpret_tensor(buf155, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf155  # reuse
        # Source Nodes: [x_272, x_276, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf156, buf153, primals_341, primals_342, primals_77, primals_78, buf154, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_78
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 224, 1, 1), (224, 1, 1, 1))
        buf158 = buf157; del buf157  # reuse
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf158, primals_209, 896, grid=grid(896), stream=stream0)
        del primals_209
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 896, 1, 1), (896, 1, 1, 1))
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf160, primals_211, 3584, grid=grid(3584), stream=stream0)
        del primals_211
        buf161 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11, x_277], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf154, buf160, buf161, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf163 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_12, x_279, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf162, primals_343, primals_344, primals_79, primals_80, buf150, buf163, 702464, grid=grid(702464), stream=stream0)
        del primals_80
        # Source Nodes: [x_287], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf165 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_288, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf164, primals_345, primals_346, primals_81, primals_82, buf165, 702464, grid=grid(702464), stream=stream0)
        del primals_82
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf166, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf167 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf169 = reinterpret_tensor(buf168, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf168  # reuse
        # Source Nodes: [x_294, x_298, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf169, buf166, primals_347, primals_348, primals_83, primals_84, buf167, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_84
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 224, 1, 1), (224, 1, 1, 1))
        buf171 = buf170; del buf170  # reuse
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf171, primals_216, 896, grid=grid(896), stream=stream0)
        del primals_216
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 896, 1, 1), (896, 1, 1, 1))
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf173, primals_218, 3584, grid=grid(3584), stream=stream0)
        del primals_218
        buf174 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12, x_299], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf167, buf173, buf174, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_300], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf176 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_13, x_301, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf175, primals_349, primals_350, primals_85, primals_86, buf163, buf176, 702464, grid=grid(702464), stream=stream0)
        del primals_86
        # Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf178 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf177, primals_351, primals_352, primals_87, primals_88, buf178, 702464, grid=grid(702464), stream=stream0)
        del primals_88
        # Source Nodes: [x_315], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf179, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf180 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf181 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf182 = reinterpret_tensor(buf181, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf181  # reuse
        # Source Nodes: [x_316, x_320, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf182, buf179, primals_353, primals_354, primals_89, primals_90, buf180, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_90
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 224, 1, 1), (224, 1, 1, 1))
        buf184 = buf183; del buf183  # reuse
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf184, primals_223, 896, grid=grid(896), stream=stream0)
        del primals_223
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 896, 1, 1), (896, 1, 1, 1))
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf186, primals_225, 3584, grid=grid(3584), stream=stream0)
        del primals_225
        buf187 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_13, x_321], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf180, buf186, buf187, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_322], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf189 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_14, x_323, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf188, primals_355, primals_356, primals_91, primals_92, buf176, buf189, 702464, grid=grid(702464), stream=stream0)
        del primals_92
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf191 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332, x_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf190, primals_357, primals_358, primals_93, primals_94, buf191, 702464, grid=grid(702464), stream=stream0)
        del primals_94
        # Source Nodes: [x_337], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf192, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf193 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf195 = reinterpret_tensor(buf194, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf194  # reuse
        # Source Nodes: [x_338, x_342, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf195, buf192, primals_359, primals_360, primals_95, primals_96, buf193, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_96
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 224, 1, 1), (224, 1, 1, 1))
        buf197 = buf196; del buf196  # reuse
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf197, primals_230, 896, grid=grid(896), stream=stream0)
        del primals_230
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 896, 1, 1), (896, 1, 1, 1))
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf199, primals_232, 3584, grid=grid(3584), stream=stream0)
        del primals_232
        buf200 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_14, x_343], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf193, buf199, buf200, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_344], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf202 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_15, x_345, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf201, primals_361, primals_362, primals_97, primals_98, buf189, buf202, 702464, grid=grid(702464), stream=stream0)
        del primals_98
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf204 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf203, primals_363, primals_364, primals_99, primals_100, buf204, 702464, grid=grid(702464), stream=stream0)
        del primals_100
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf205, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf206 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf207 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf208 = reinterpret_tensor(buf207, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf207  # reuse
        # Source Nodes: [x_360, x_364, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf208, buf205, primals_365, primals_366, primals_101, primals_102, buf206, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_102
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 224, 1, 1), (224, 1, 1, 1))
        buf210 = buf209; del buf209  # reuse
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf210, primals_237, 896, grid=grid(896), stream=stream0)
        del primals_237
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 896, 1, 1), (896, 1, 1, 1))
        buf212 = buf211; del buf211  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf212, primals_239, 3584, grid=grid(3584), stream=stream0)
        del primals_239
        buf213 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_15, x_365], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf206, buf212, buf213, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_366], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf215 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_16, x_367, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf214, primals_367, primals_368, primals_103, primals_104, buf202, buf215, 702464, grid=grid(702464), stream=stream0)
        del primals_104
        # Source Nodes: [x_375], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf217 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf216, primals_369, primals_370, primals_105, primals_106, buf217, 702464, grid=grid(702464), stream=stream0)
        del primals_106
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf218, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf219 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf221 = reinterpret_tensor(buf220, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf220  # reuse
        # Source Nodes: [x_382, x_386, x_se_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf221, buf218, primals_371, primals_372, primals_107, primals_108, buf219, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_108
        # Source Nodes: [x_se_65], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 224, 1, 1), (224, 1, 1, 1))
        buf223 = buf222; del buf222  # reuse
        # Source Nodes: [x_se_65, x_se_66], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf223, primals_244, 896, grid=grid(896), stream=stream0)
        del primals_244
        # Source Nodes: [x_se_67], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_245, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 896, 1, 1), (896, 1, 1, 1))
        buf225 = buf224; del buf224  # reuse
        # Source Nodes: [x_se_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf225, primals_246, 3584, grid=grid(3584), stream=stream0)
        del primals_246
        buf226 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_16, x_387], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf219, buf225, buf226, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_388], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf228 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17, x_389, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf227, primals_373, primals_374, primals_109, primals_110, buf215, buf228, 702464, grid=grid(702464), stream=stream0)
        del primals_110
        # Source Nodes: [x_397], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf230 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398, x_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf229, primals_375, primals_376, primals_111, primals_112, buf230, 702464, grid=grid(702464), stream=stream0)
        del primals_112
        # Source Nodes: [x_403], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf231, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf232 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf233 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf234 = reinterpret_tensor(buf233, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf233  # reuse
        # Source Nodes: [x_404, x_408, x_se_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf234, buf231, primals_377, primals_378, primals_113, primals_114, buf232, 3584, 196, grid=grid(3584), stream=stream0)
        del primals_114
        # Source Nodes: [x_se_69], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 224, 1, 1), (224, 1, 1, 1))
        buf236 = buf235; del buf235  # reuse
        # Source Nodes: [x_se_69, x_se_70], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf236, primals_251, 896, grid=grid(896), stream=stream0)
        del primals_251
        # Source Nodes: [x_se_71], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 896, 1, 1), (896, 1, 1, 1))
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [x_se_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf238, primals_253, 3584, grid=grid(3584), stream=stream0)
        del primals_253
        buf239 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_17, x_409], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_21.run(buf232, buf238, buf239, 702464, grid=grid(702464), stream=stream0)
        # Source Nodes: [x_410], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 896, 14, 14), (175616, 196, 14, 1))
        buf241 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_18, x_411, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf240, primals_379, primals_380, primals_115, primals_116, buf228, buf241, 702464, grid=grid(702464), stream=stream0)
        del primals_116
        # Source Nodes: [x_420], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 2240, 14, 14), (439040, 196, 14, 1))
        buf243 = empty((4, 2240, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_421, x_425], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf242, primals_381, primals_382, primals_117, primals_118, buf243, 1756160, grid=grid(1756160), stream=stream0)
        del primals_118
        # Source Nodes: [x_426], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_256, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf244, (4, 2240, 7, 7), (109760, 49, 7, 1))
        buf245 = empty((4, 2240, 7, 7), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((4, 2240, 1, 1), (2240, 1, 8960, 8960), device='cuda', dtype=torch.float32)
        buf247 = reinterpret_tensor(buf246, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf246  # reuse
        # Source Nodes: [x_427, x_431, x_se_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27.run(buf247, buf244, primals_383, primals_384, primals_119, primals_120, buf245, 8960, 49, grid=grid(8960), stream=stream0)
        del primals_120
        # Source Nodes: [x_se_73], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 224, 1, 1), (224, 1, 1, 1))
        buf249 = buf248; del buf248  # reuse
        # Source Nodes: [x_se_73, x_se_74], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf249, primals_258, 896, grid=grid(896), stream=stream0)
        del primals_258
        # Source Nodes: [x_se_75], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 2240, 1, 1), (2240, 1, 1, 1))
        buf251 = buf250; del buf250  # reuse
        # Source Nodes: [x_se_75], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf251, primals_260, 8960, grid=grid(8960), stream=stream0)
        del primals_260
        buf252 = empty((4, 2240, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_18, x_432], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_29.run(buf245, buf251, buf252, 439040, grid=grid(439040), stream=stream0)
        # Source Nodes: [x_433], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 2240, 7, 7), (109760, 49, 7, 1))
        # Source Nodes: [x_439], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf241, primals_262, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 2240, 7, 7), (109760, 49, 7, 1))
        buf259 = empty((4, 2240, 7, 7), device='cuda', dtype=torch.bool)
        buf256 = empty_strided((4, 2240, 1, 1), (2240, 1, 8960, 8960), device='cuda', dtype=torch.float32)
        buf257 = reinterpret_tensor(buf256, (4, 2240), (2240, 1), 0); del buf256  # reuse
        # Source Nodes: [x_434, x_440, x_444, x_447, x_450, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_30.run(buf257, buf253, primals_385, primals_386, primals_121, primals_122, buf254, primals_387, primals_388, primals_123, primals_124, buf259, 8960, 49, grid=grid(8960), stream=stream0)
        del primals_122
        del primals_124
        buf258 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_264, buf257, reinterpret_tensor(primals_263, (2240, 1000), (1, 2240), 0), alpha=1, beta=1, out=buf258)
        del primals_264
        return (buf258, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, buf0, buf1, buf2, buf3, buf4, buf5, buf7, buf9, buf11, buf12, buf13, buf14, buf16, buf17, buf18, buf19, buf20, buf22, buf24, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf35, buf37, buf39, buf40, buf41, buf42, buf44, buf45, buf46, buf47, buf48, buf50, buf52, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf63, buf65, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf76, buf78, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf89, buf91, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf102, buf104, buf106, buf107, buf108, buf109, buf111, buf112, buf113, buf114, buf115, buf117, buf119, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf130, buf132, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf143, buf145, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf156, buf158, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf169, buf171, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf182, buf184, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf195, buf197, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf208, buf210, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf221, buf223, buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf234, buf236, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf247, buf249, buf251, buf252, buf253, buf254, buf257, reinterpret_tensor(primals_263, (1000, 2240), (2240, 1), 0), buf259, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((8, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((224, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((56, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((224, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((56, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((448, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((112, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((896, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((2240, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((224, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2240, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((2240, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1000, 2240), (2240, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_regnet', benchmark_compiled_module)
