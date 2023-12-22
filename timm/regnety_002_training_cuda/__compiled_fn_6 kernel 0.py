
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/it/citjxewdi5uiiczpwndabh5a4ogowsqnc6gp2dwqpaqemqdfihoo.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (368*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtnt3ctnn6cl54lyaq533h7whnvpf5cqhru3ru4uoudqyijnhnh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxopeduxjzns75e47piclesk27p33b5i3zstc5pafd7koxpnnbcs.py
# Source Nodes: [sigmoid_12], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_12 => sigmoid_12
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hm/chmue5p3boblvzbssqaaso47gjdgdor6uxzcp5prhokpoa3hsitq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 368
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (368*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvbnc2janob2kxzeqjujpwpoklqe7nctpxb2ghtjlgladf3trvt.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmyp25jpbp7tycrh65nrjfvfkspgjar4byucpehfoqsnath6e3r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 92
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (92*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcbedt4la655duutnliyc7gkjg4c2bf3jq3dj4p6hljvkzvkja4.py
# Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_12 => sigmoid_12
triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (368*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + (368*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 49.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = tmp16 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = tmp23 * tmp24
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkgym4qlyvw4byoj2ykgzc7vd7bjegigc4zxyts2bdzkecmjygt.py
# Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_12 => sigmoid_12
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 49.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 0.002551020408163265
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coeq6fch7n57ca3s3khjpax5xqmlws6ojxppow4rjfzum53jdbbn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtgzbwzahe2fvj2v3zgvcsud76uaiqi4mvu3b2jpwwf33z5ls6u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv7lxmvnkfenihpju5b7mut4iqjd5u3qsu7rpysbkvc67wmcxuz.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + (368*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwgojhkkgpu4a26a6t7h6t7dc7yz3k4tltw3chhmsqkryqeimgf.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x3), xmask)
    tmp11 = tl.load(in_ptr4 + (x3), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.002551020408163265
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crgmcxht4hdrpitoozifvsi7gentrkeeyfgm7h6c5i2q4f25aoum.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x2), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxlfbxdkdo52qm3fdmmhw4j3vlvdu5v2k3byqkeecnqb4eglbsz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa7b7hwlo7taa7aqcr3tqymb4yfkuvhu5u5x3k7mq6pibgcr3mr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/coutsxuptl7ckcfo6dqrl6s6ujqkzf7kgrx2q2bhk462itmcjzed.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl72w2kgiin4ozmertyztzu5zjgu65a26jyhypolndqbov2ih3tt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fjqph6dvky7hjaiiddly7vbv5if4aafv6toygnzwwme6zsiuyk.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvpxbjao5xfbhpte3jxve6vvvjiy6t7trpzszeuklvsvauwax7b.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63gx5vzhcu2ejnixn5rdetasixz3f5jfbgszrevamhgcf7dctt3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp12 * tmp21
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr4 + (x0), tmp24, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csg3ciwlpgpdc75g7ja2zoht75mjq3pabd3d4o6rtelocphtoh5f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5c7apw23pna46f6w7wqflob7klxwj2xsen7zv4jwnljgbj5lsy.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2fsn6buwqb3e3vunm7yopb4sn4pziurkbdbrrd3dsytrkmtlk3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 38
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (38*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcjbgkajtxkncve3ftna6ytgdf527qerfa3l74lf5oazyfem6ty.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 368
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (72128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (72128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (72128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujaxterslfjvoc6kyu72gapuvpsc2qz5ug5hzqyj6s4ijet5ek3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cld7dtosqv6zepmmxauavty4t5zqo47sifo5moig5wldwwh2ykay.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cno3nmfm5qwtckivlxmbthvnsiezskjd3hwcaw7o34nsk2nv2c3d.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxjg7cq4snwrh5ktavvlregslmcjqemr23i4f2vuy7devesciglf.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_5 => sigmoid_5
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6a4litc7tmmxoe3he4nlqzn2zmkgrjnehgpitbxeyogwgv7hoyg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ji/cjietpmpxc52hnmvwtivypgrjqje5s27s4jj2sk5mczmrqbi7z4c.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_5 => sigmoid_5
triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (152*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (152*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndiior2c6kt3jujcyrpawfqmi4g5wgtbhxjvfjcrxzytddgyk7e.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_5 => sigmoid_5
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 196.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 0.0006377551020408163
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicjnpseprvecf2uzni5ygcftuxb633th7yibps63huxtdi2xdn6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74sh2ksnlbnw6bvp3bkw2mru42aiyclgbpfmrcf7mfdhn2jq7so.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymdodgxx6tl7ayr6drsd6snnhzxg7xrv64qkay4w7iag3nnl4yy.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32o452jyryvkslut7romejng6sirbxnpg27gvsqjidgvu4cezo6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch52fugwgwjtgnujr4b7oxy5nbt6v6z6effeudjkr7ugky6nbdvv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2c46ucawzwox4rkpkhujpf2qqsqw5g7favclraiqjgtitkmumwl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tmp9 * tmp18
    tmp21 = tmp16 * tmp20
    tl.store(out_ptr3 + (x0), tmp19, xmask)
    tl.store(out_ptr4 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmapf4srlnfe4tqmpp7sriv7dztaisaereqpsku4ynkz6ye6g5v4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coexlade7sa75nodyz32gotki73byyxjyz6j2xqlhdw5yzrg5awj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4n4ajfy2omvkpvn3h6bnbjmab3i2ytacmkcyq7t3qwk42b4fkp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (14*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3tqsa3txnvsdejec6kzpgmbl2mzgdguf45eo5rubzgtmnewrpo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4ndjpvxsnhnmimknwtuu22dkhpar7tral2nt62qilzzqwyz5jc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5dcab5k3xbyighuvqmmy6ibnayxmoy47355on237cb3l26ixp5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp22, xmask)
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tmp15 * tmp24
    tmp27 = tmp22 * tmp26
    tl.store(out_ptr3 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/coug4zdoxyll47kgkatuha4y7v5ddbfn6brpfbkx3h7kaasd2iaj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x3), xmask)
    tmp25 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tl.store(out_ptr1 + (x3), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/cat3diowuzcdjmfijtjeqm7mxellgr36wsgnpraefku2wyabzq6q.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_1 => sigmoid_1
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    xnumel = 448
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/canwiqw4nuf2j5x5diwty7qkksimmj74lfvnzob2a4yie2cdnjfb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmgucc3o2rd4wzn4tb3dgn4mfzlsqbzvsnuoq2gju5uzv5qgrpq.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprxf3za724zpifpaqw4u4kszeytkdfkmkzswckr32qvyfoiu74y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxnw35tayyzhoojq6uuu4nq5uxo746bkoadbw6wzey7crqjobam.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_1 => sigmoid_1
triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (56*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (56*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 784.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vktnp6jrjufly6q5usjmb4glshcc57ovku4wanarxv65s4uktg.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_1 => sigmoid_1
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 784.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 0.00015943877551020407
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7aapnvijsyevfry4y2bvvbre2nrsidyvkfrwp4mck336u55rku.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (175616*(r2 // 3136)) + (351232*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (175616*(r2 // 3136)) + (351232*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (175616*(r2 // 3136)) + (351232*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdptqsjyvqveh55pg37lf66ed63cuysal3mbndwajcvdaspxi2pg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbaw25p2j33vlq3cuz6hrvo7sryvfxrd2527mqncedjw65iofpl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5f4qno2ngnnxz2hbzcjty4ykipghkaelre6yg6xnullhilthxe.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 56
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwrue3rrzhs7vb54s6opkotq56jarohrnee253wnawlux33pepz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctl4tjhailb56hxqh66q3y4egq7dctivm6ks257joaorkotne5fu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jd65wsnvdgybanonbpkbv2a7cyg5bzsoasv6ifea5tjtvfyykg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfb4jbd6eqam2jn3f7wv7mebrt3okflg4yuwxdiqd3e6mefj63a.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x3), None)
    tmp25 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x3), tmp23, None)
    tl.store(out_ptr1 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nwwb6lgeprcljd5geyrk4besbgsrzbyu5jw6uohugmsdrof45x.py
# Source Nodes: [sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid => sigmoid
triton_red_fused_mul_sigmoid_sigmoid_backward_sum_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sigmoid_backward_sum_59', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crz4kzxq3femvg32svejs7uq44ssq5tzubzwgmivult7zdgt5vbb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chv4owinerk4krsavagrmclo25hcxbwm754vgfahlgqlkfixtgko.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7a7nwomqlupwle6si3xozyf6b7iwxw3cjuv7dvx22mr3nydnpm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ji/cjisjpuhonbtjaun5d3j74zrescky36w447mpkw6n2vwez5dnevr.py
# Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid => sigmoid
triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (24*(r2 // 3136)) + (48*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (24*(r2 // 3136)) + (48*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 3136.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6krxgceomrtlzzmxse4podnqjklotiaqwurhrkgvyl6d4eq44mz.py
# Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid => sigmoid
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 3136.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 3.985969387755102e-05
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pvjjbf7g6ks7zvgycy6vbmydg3j2xsqpg4k43zmjogu42smr6c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 312
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (301056*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (301056*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + ((12544*x1) + (301056*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2s4pm2pxpnpoagjlz5gdkccqhrkowa53qakqiydfbk3jlpvaxe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7md3py7jsctvwddxt6d3rgjoocwo3ztwqi5gy3lwgkdea6lch7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2dgqp2mpkfdaoo3htar2wbsjbu462n3faaypvgnscbfrbm23r3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5juh3shhluljfnb466gpbhgqyuhfdephjeomxayeitcqqh6dvjo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr3 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp9 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crrxvxflsiljlmbgdflxtmuom7jnc5imbcjk5wrse4ydu7sp6nkd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nj5njnld7dbjj6tbmxh2wijnypyuc4xh7atjnn4cb223dr6vl3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg52saxgjuk72aukdcapttf3zvt65b3nhqzbus373uxecvqmlp6k.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, squeeze_19, relu_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, convolution_12, squeeze_25, relu_8, convolution_13, squeeze_28, relu_9, convolution_14, squeeze_31, relu_10, mean_2, relu_11, convolution_16, mul_79, convolution_17, squeeze_34, convolution_18, squeeze_37, relu_12, convolution_19, squeeze_40, relu_13, convolution_20, squeeze_43, relu_14, mean_3, relu_15, convolution_22, mul_108, convolution_23, squeeze_46, relu_16, convolution_24, squeeze_49, relu_17, convolution_25, squeeze_52, relu_18, mean_4, relu_19, convolution_27, mul_130, convolution_28, squeeze_55, relu_20, convolution_29, squeeze_58, relu_21, convolution_30, squeeze_61, relu_22, mean_5, relu_23, convolution_32, mul_152, convolution_33, squeeze_64, relu_24, convolution_34, squeeze_67, relu_25, convolution_35, squeeze_70, relu_26, mean_6, relu_27, convolution_37, mul_174, convolution_38, squeeze_73, convolution_39, squeeze_76, relu_28, convolution_40, squeeze_79, relu_29, convolution_41, squeeze_82, relu_30, mean_7, relu_31, convolution_43, mul_203, convolution_44, squeeze_85, relu_32, convolution_45, squeeze_88, relu_33, convolution_46, squeeze_91, relu_34, mean_8, relu_35, convolution_48, mul_225, convolution_49, squeeze_94, relu_36, convolution_50, squeeze_97, relu_37, convolution_51, squeeze_100, relu_38, mean_9, relu_39, convolution_53, mul_247, convolution_54, squeeze_103, relu_40, convolution_55, squeeze_106, relu_41, convolution_56, squeeze_109, relu_42, mean_10, relu_43, convolution_58, mul_269, convolution_59, squeeze_112, relu_44, convolution_60, squeeze_115, relu_45, convolution_61, squeeze_118, relu_46, mean_11, relu_47, convolution_63, mul_291, convolution_64, squeeze_121, relu_48, convolution_65, squeeze_124, relu_49, convolution_66, squeeze_127, relu_50, mean_12, relu_51, convolution_68, mul_313, convolution_69, squeeze_130, clone, permute_1, le, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_11, (56, ), (1, ))
    assert_size_stride(primals_13, (56, ), (1, ))
    assert_size_stride(primals_15, (56, ), (1, ))
    assert_size_stride(primals_17, (56, ), (1, ))
    assert_size_stride(primals_19, (152, ), (1, ))
    assert_size_stride(primals_21, (152, ), (1, ))
    assert_size_stride(primals_23, (152, ), (1, ))
    assert_size_stride(primals_25, (152, ), (1, ))
    assert_size_stride(primals_27, (152, ), (1, ))
    assert_size_stride(primals_29, (152, ), (1, ))
    assert_size_stride(primals_31, (152, ), (1, ))
    assert_size_stride(primals_33, (152, ), (1, ))
    assert_size_stride(primals_35, (152, ), (1, ))
    assert_size_stride(primals_37, (152, ), (1, ))
    assert_size_stride(primals_39, (152, ), (1, ))
    assert_size_stride(primals_41, (152, ), (1, ))
    assert_size_stride(primals_43, (152, ), (1, ))
    assert_size_stride(primals_45, (368, ), (1, ))
    assert_size_stride(primals_47, (368, ), (1, ))
    assert_size_stride(primals_49, (368, ), (1, ))
    assert_size_stride(primals_51, (368, ), (1, ))
    assert_size_stride(primals_53, (368, ), (1, ))
    assert_size_stride(primals_55, (368, ), (1, ))
    assert_size_stride(primals_57, (368, ), (1, ))
    assert_size_stride(primals_59, (368, ), (1, ))
    assert_size_stride(primals_61, (368, ), (1, ))
    assert_size_stride(primals_63, (368, ), (1, ))
    assert_size_stride(primals_65, (368, ), (1, ))
    assert_size_stride(primals_67, (368, ), (1, ))
    assert_size_stride(primals_69, (368, ), (1, ))
    assert_size_stride(primals_71, (368, ), (1, ))
    assert_size_stride(primals_73, (368, ), (1, ))
    assert_size_stride(primals_75, (368, ), (1, ))
    assert_size_stride(primals_77, (368, ), (1, ))
    assert_size_stride(primals_79, (368, ), (1, ))
    assert_size_stride(primals_81, (368, ), (1, ))
    assert_size_stride(primals_83, (368, ), (1, ))
    assert_size_stride(primals_85, (368, ), (1, ))
    assert_size_stride(primals_87, (368, ), (1, ))
    assert_size_stride(primals_89, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_90, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_91, (24, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_92, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_94, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_96, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_97, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_98, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_99, (56, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_100, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_102, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_104, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_105, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_106, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_107, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_108, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_110, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_112, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_113, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_114, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_115, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_116, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_118, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_120, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_121, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_122, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_123, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_125, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_127, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_128, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_129, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_130, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_132, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_134, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_135, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_136, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_137, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_139, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_141, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_142, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_143, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_144, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_145, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_147, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_149, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_150, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_151, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_152, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_154, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_156, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_157, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_158, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_159, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_161, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_163, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_164, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_165, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_166, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_168, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_170, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_171, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_172, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_173, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_175, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_177, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_178, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_179, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_180, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_182, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_184, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_319, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_1, (8, 24, 112, 112), (301056, 12544, 112, 1))
    assert_size_stride(squeeze_4, (24, ), (1, ))
    assert_size_stride(relu_1, (8, 24, 112, 112), (301056, 12544, 112, 1))
    assert_size_stride(convolution_2, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(squeeze_7, (24, ), (1, ))
    assert_size_stride(relu_2, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(mean, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(relu_3, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(convolution_4, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(mul_21, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(squeeze_10, (24, ), (1, ))
    assert_size_stride(convolution_6, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(squeeze_13, (24, ), (1, ))
    assert_size_stride(relu_4, (8, 24, 56, 56), (75264, 3136, 56, 1))
    assert_size_stride(convolution_7, (8, 56, 56, 56), (175616, 3136, 56, 1))
    assert_size_stride(squeeze_16, (56, ), (1, ))
    assert_size_stride(relu_5, (8, 56, 56, 56), (175616, 3136, 56, 1))
    assert_size_stride(convolution_8, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(squeeze_19, (56, ), (1, ))
    assert_size_stride(relu_6, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(mean_1, (8, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(relu_7, (8, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(convolution_10, (8, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(mul_50, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(convolution_11, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(squeeze_22, (56, ), (1, ))
    assert_size_stride(convolution_12, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(squeeze_25, (56, ), (1, ))
    assert_size_stride(relu_8, (8, 56, 28, 28), (43904, 784, 28, 1))
    assert_size_stride(convolution_13, (8, 152, 28, 28), (119168, 784, 28, 1))
    assert_size_stride(squeeze_28, (152, ), (1, ))
    assert_size_stride(relu_9, (8, 152, 28, 28), (119168, 784, 28, 1))
    assert_size_stride(convolution_14, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_31, (152, ), (1, ))
    assert_size_stride(relu_10, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(mean_2, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(relu_11, (8, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(convolution_16, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(mul_79, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_17, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_34, (152, ), (1, ))
    assert_size_stride(convolution_18, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_37, (152, ), (1, ))
    assert_size_stride(relu_12, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_19, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_40, (152, ), (1, ))
    assert_size_stride(relu_13, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_20, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_43, (152, ), (1, ))
    assert_size_stride(relu_14, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(mean_3, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(relu_15, (8, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(convolution_22, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(mul_108, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_23, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_46, (152, ), (1, ))
    assert_size_stride(relu_16, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_24, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_49, (152, ), (1, ))
    assert_size_stride(relu_17, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_25, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_52, (152, ), (1, ))
    assert_size_stride(relu_18, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(mean_4, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(relu_19, (8, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(convolution_27, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(mul_130, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_28, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_55, (152, ), (1, ))
    assert_size_stride(relu_20, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_29, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_58, (152, ), (1, ))
    assert_size_stride(relu_21, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_30, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_61, (152, ), (1, ))
    assert_size_stride(relu_22, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(mean_5, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(relu_23, (8, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(convolution_32, (8, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(mul_152, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_33, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_64, (152, ), (1, ))
    assert_size_stride(relu_24, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_34, (8, 368, 14, 14), (72128, 196, 14, 1))
    assert_size_stride(squeeze_67, (368, ), (1, ))
    assert_size_stride(relu_25, (8, 368, 14, 14), (72128, 196, 14, 1))
    assert_size_stride(convolution_35, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_70, (368, ), (1, ))
    assert_size_stride(relu_26, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_6, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_27, (8, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(convolution_37, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_174, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_38, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_73, (368, ), (1, ))
    assert_size_stride(convolution_39, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_76, (368, ), (1, ))
    assert_size_stride(relu_28, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_40, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_79, (368, ), (1, ))
    assert_size_stride(relu_29, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_41, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_82, (368, ), (1, ))
    assert_size_stride(relu_30, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_7, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_31, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_43, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_203, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_44, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_85, (368, ), (1, ))
    assert_size_stride(relu_32, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_45, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_88, (368, ), (1, ))
    assert_size_stride(relu_33, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_46, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_91, (368, ), (1, ))
    assert_size_stride(relu_34, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_8, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_35, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_48, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_225, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_49, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_94, (368, ), (1, ))
    assert_size_stride(relu_36, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_50, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_97, (368, ), (1, ))
    assert_size_stride(relu_37, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_51, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_100, (368, ), (1, ))
    assert_size_stride(relu_38, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_9, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_39, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_53, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_247, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_54, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_103, (368, ), (1, ))
    assert_size_stride(relu_40, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_55, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_106, (368, ), (1, ))
    assert_size_stride(relu_41, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_56, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_109, (368, ), (1, ))
    assert_size_stride(relu_42, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_10, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_43, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_58, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_269, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_59, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_112, (368, ), (1, ))
    assert_size_stride(relu_44, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_60, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_115, (368, ), (1, ))
    assert_size_stride(relu_45, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_61, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_118, (368, ), (1, ))
    assert_size_stride(relu_46, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_11, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_47, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_63, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_291, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_64, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_121, (368, ), (1, ))
    assert_size_stride(relu_48, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_65, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_124, (368, ), (1, ))
    assert_size_stride(relu_49, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_66, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_127, (368, ), (1, ))
    assert_size_stride(relu_50, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(mean_12, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(relu_51, (8, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(convolution_68, (8, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(mul_313, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(convolution_69, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(squeeze_130, (368, ), (1, ))
    assert_size_stride(clone, (8, 368), (368, 1))
    assert_size_stride(permute_1, (1000, 368), (368, 1))
    assert_size_stride(le, (8, 368, 7, 7), (18032, 49, 7, 1))
    assert_size_stride(unsqueeze_178, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 368), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 368), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((368, ), device='cuda', dtype=torch.float32)
        buf4 = empty((368, ), device='cuda', dtype=torch.float32)
        buf5 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_69, unsqueeze_178, squeeze_130, buf3, buf4, buf5, 368, 392, grid=grid(368), stream=stream0)
        buf6 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, convolution_69, unsqueeze_178, buf4, squeeze_130, buf3, primals_87, buf6, 144256, grid=grid(144256), stream=stream0)
        del convolution_69
        del primals_87
        del squeeze_130
        del unsqueeze_178
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, mul_313, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del mul_313
        del primals_184
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf11 = reinterpret_tensor(buf10, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf10  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf11, buf8, relu_50, convolution_68, 2944, 49, grid=grid(2944), stream=stream0)
        buf12 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf11, buf12, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf13 = aten.convolution_backward(buf11, relu_51, primals_182, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf11
        del primals_182
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf16, relu_51, 736, grid=grid(736), stream=stream0)
        del relu_51
        buf17 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf16, buf17, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf18 = aten.convolution_backward(buf16, mean_12, primals_180, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf16
        del mean_12
        del primals_180
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = empty((368, ), device='cuda', dtype=torch.float32)
        buf22 = empty((368, ), device='cuda', dtype=torch.float32)
        buf24 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_50, buf8, convolution_68, buf19, convolution_66, unsqueeze_190, squeeze_127, buf21, buf22, buf24, 368, 392, grid=grid(368), stream=stream0)
        buf23 = buf8; del buf8  # reuse
        buf25 = buf23; del buf23  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf25, relu_50, convolution_68, buf19, convolution_66, unsqueeze_190, buf22, squeeze_127, buf21, primals_85, 144256, grid=grid(144256), stream=stream0)
        del convolution_66
        del convolution_68
        del primals_85
        del relu_50
        del squeeze_127
        del unsqueeze_190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf26 = aten.convolution_backward(buf25, relu_49, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf25
        del primals_179
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf22; del buf22  # reuse
        buf30 = empty((368, ), device='cuda', dtype=torch.float32)
        buf31 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_49, buf27, convolution_65, unsqueeze_202, squeeze_124, buf29, buf30, buf31, 368, 392, grid=grid(368), stream=stream0)
        buf32 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf32, relu_49, convolution_65, unsqueeze_202, buf30, squeeze_124, buf29, primals_83, 144256, grid=grid(144256), stream=stream0)
        del convolution_65
        del primals_83
        del relu_49
        del squeeze_124
        del unsqueeze_202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf33 = aten.convolution_backward(buf32, relu_48, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_178
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf36 = buf30; del buf30  # reuse
        buf37 = empty((368, ), device='cuda', dtype=torch.float32)
        buf39 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_11.run(relu_48, le, buf0, buf34, convolution_64, unsqueeze_214, squeeze_121, buf36, buf37, buf39, 368, 392, grid=grid(368), stream=stream0)
        buf38 = buf32; del buf32  # reuse
        buf40 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12.run(buf40, relu_48, le, buf0, buf34, convolution_64, unsqueeze_214, buf37, squeeze_121, buf36, primals_81, 144256, grid=grid(144256), stream=stream0)
        del convolution_64
        del primals_81
        del squeeze_121
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf41 = aten.convolution_backward(buf40, mul_291, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf40
        del mul_291
        del primals_177
        buf42 = buf41[0]
        buf43 = buf41[1]
        del buf41
        buf44 = reinterpret_tensor(buf19, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf19  # reuse
        buf45 = reinterpret_tensor(buf44, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf44  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf45, buf42, relu_46, convolution_63, 2944, 49, grid=grid(2944), stream=stream0)
        buf46 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf45, buf46, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf47 = aten.convolution_backward(buf45, relu_47, primals_175, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf45
        del primals_175
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        buf50 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf50, relu_47, 736, grid=grid(736), stream=stream0)
        del relu_47
        buf51 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf50, buf51, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf52 = aten.convolution_backward(buf50, mean_11, primals_173, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf50
        del mean_11
        del primals_173
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = empty((368, ), device='cuda', dtype=torch.float32)
        buf56 = empty((368, ), device='cuda', dtype=torch.float32)
        buf58 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_46, buf42, convolution_63, buf53, convolution_61, unsqueeze_226, squeeze_118, buf55, buf56, buf58, 368, 392, grid=grid(368), stream=stream0)
        buf57 = buf42; del buf42  # reuse
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf59, relu_46, convolution_63, buf53, convolution_61, unsqueeze_226, buf56, squeeze_118, buf55, primals_79, 144256, grid=grid(144256), stream=stream0)
        del buf53
        del convolution_61
        del convolution_63
        del primals_79
        del relu_46
        del squeeze_118
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf60 = aten.convolution_backward(buf59, relu_45, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf59
        del primals_172
        buf61 = buf60[0]
        buf62 = buf60[1]
        del buf60
        buf63 = buf56; del buf56  # reuse
        buf64 = empty((368, ), device='cuda', dtype=torch.float32)
        buf65 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_45, buf61, convolution_60, unsqueeze_238, squeeze_115, buf63, buf64, buf65, 368, 392, grid=grid(368), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf66, relu_45, convolution_60, unsqueeze_238, buf64, squeeze_115, buf63, primals_77, 144256, grid=grid(144256), stream=stream0)
        del convolution_60
        del primals_77
        del relu_45
        del squeeze_115
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf67 = aten.convolution_backward(buf66, relu_44, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf66
        del primals_171
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_13.run(buf70, relu_44, relu_48, le, buf0, buf68, 144256, grid=grid(144256), stream=stream0)
        del le
        del relu_44
        del relu_48
        buf71 = buf64; del buf64  # reuse
        buf72 = empty((368, ), device='cuda', dtype=torch.float32)
        buf73 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf70, convolution_59, unsqueeze_250, squeeze_112, buf71, buf72, buf73, 368, 392, grid=grid(368), stream=stream0)
        buf74 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_15.run(buf70, convolution_59, unsqueeze_250, buf72, squeeze_112, buf71, primals_75, buf74, 144256, grid=grid(144256), stream=stream0)
        del convolution_59
        del primals_75
        del squeeze_112
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf75 = aten.convolution_backward(buf74, mul_269, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf74
        del mul_269
        del primals_170
        buf76 = buf75[0]
        buf77 = buf75[1]
        del buf75
        buf78 = reinterpret_tensor(buf0, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf0  # reuse
        buf79 = reinterpret_tensor(buf78, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf78  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf79, buf76, relu_42, convolution_58, 2944, 49, grid=grid(2944), stream=stream0)
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf79, buf80, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf81 = aten.convolution_backward(buf79, relu_43, primals_168, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf79
        del primals_168
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        buf84 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf84, relu_43, 736, grid=grid(736), stream=stream0)
        del relu_43
        buf85 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf84, buf85, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf86 = aten.convolution_backward(buf84, mean_10, primals_166, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf84
        del mean_10
        del primals_166
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        buf89 = empty((368, ), device='cuda', dtype=torch.float32)
        buf90 = empty((368, ), device='cuda', dtype=torch.float32)
        buf92 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_42, buf76, convolution_58, buf87, convolution_56, unsqueeze_262, squeeze_109, buf89, buf90, buf92, 368, 392, grid=grid(368), stream=stream0)
        buf91 = buf76; del buf76  # reuse
        buf93 = buf91; del buf91  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf93, relu_42, convolution_58, buf87, convolution_56, unsqueeze_262, buf90, squeeze_109, buf89, primals_73, 144256, grid=grid(144256), stream=stream0)
        del convolution_56
        del convolution_58
        del primals_73
        del relu_42
        del squeeze_109
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf94 = aten.convolution_backward(buf93, relu_41, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf93
        del primals_165
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = buf90; del buf90  # reuse
        buf98 = empty((368, ), device='cuda', dtype=torch.float32)
        buf99 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_41, buf95, convolution_55, unsqueeze_274, squeeze_106, buf97, buf98, buf99, 368, 392, grid=grid(368), stream=stream0)
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf100, relu_41, convolution_55, unsqueeze_274, buf98, squeeze_106, buf97, primals_71, 144256, grid=grid(144256), stream=stream0)
        del convolution_55
        del primals_71
        del relu_41
        del squeeze_106
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf101 = aten.convolution_backward(buf100, relu_40, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_164
        buf102 = buf101[0]
        buf103 = buf101[1]
        del buf101
        buf104 = buf98; del buf98  # reuse
        buf105 = empty((368, ), device='cuda', dtype=torch.float32)
        buf107 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_16.run(relu_40, buf70, buf102, convolution_54, unsqueeze_286, squeeze_103, buf104, buf105, buf107, 368, 392, grid=grid(368), stream=stream0)
        buf106 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_17.run(relu_40, buf70, buf102, convolution_54, unsqueeze_286, buf105, squeeze_103, buf104, primals_69, buf106, 144256, grid=grid(144256), stream=stream0)
        del convolution_54
        del primals_69
        del squeeze_103
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf108 = aten.convolution_backward(buf106, mul_247, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf106
        del mul_247
        del primals_163
        buf109 = buf108[0]
        buf110 = buf108[1]
        del buf108
        buf111 = reinterpret_tensor(buf87, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf87  # reuse
        buf112 = reinterpret_tensor(buf111, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf111  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf112, buf109, relu_38, convolution_53, 2944, 49, grid=grid(2944), stream=stream0)
        buf113 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf112, buf113, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf114 = aten.convolution_backward(buf112, relu_39, primals_161, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf112
        del primals_161
        buf115 = buf114[0]
        buf116 = buf114[1]
        del buf114
        buf117 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf117, relu_39, 736, grid=grid(736), stream=stream0)
        del relu_39
        buf118 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf117, buf118, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf119 = aten.convolution_backward(buf117, mean_9, primals_159, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf117
        del mean_9
        del primals_159
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = empty((368, ), device='cuda', dtype=torch.float32)
        buf123 = empty((368, ), device='cuda', dtype=torch.float32)
        buf125 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_38, buf109, convolution_53, buf120, convolution_51, unsqueeze_298, squeeze_100, buf122, buf123, buf125, 368, 392, grid=grid(368), stream=stream0)
        buf124 = buf109; del buf109  # reuse
        buf126 = buf124; del buf124  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf126, relu_38, convolution_53, buf120, convolution_51, unsqueeze_298, buf123, squeeze_100, buf122, primals_67, 144256, grid=grid(144256), stream=stream0)
        del convolution_51
        del convolution_53
        del primals_67
        del relu_38
        del squeeze_100
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf127 = aten.convolution_backward(buf126, relu_37, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf126
        del primals_158
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf123; del buf123  # reuse
        buf131 = empty((368, ), device='cuda', dtype=torch.float32)
        buf132 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_37, buf128, convolution_50, unsqueeze_310, squeeze_97, buf130, buf131, buf132, 368, 392, grid=grid(368), stream=stream0)
        buf133 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf133, relu_37, convolution_50, unsqueeze_310, buf131, squeeze_97, buf130, primals_65, 144256, grid=grid(144256), stream=stream0)
        del convolution_50
        del primals_65
        del relu_37
        del squeeze_97
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf134 = aten.convolution_backward(buf133, relu_36, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf133
        del primals_157
        buf135 = buf134[0]
        buf136 = buf134[1]
        del buf134
        buf137 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_18.run(buf137, relu_36, relu_40, buf70, buf135, 144256, grid=grid(144256), stream=stream0)
        del buf135
        del relu_36
        del relu_40
        buf138 = buf131; del buf131  # reuse
        buf139 = empty((368, ), device='cuda', dtype=torch.float32)
        buf140 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf137, convolution_49, unsqueeze_322, squeeze_94, buf138, buf139, buf140, 368, 392, grid=grid(368), stream=stream0)
        buf141 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_15.run(buf137, convolution_49, unsqueeze_322, buf139, squeeze_94, buf138, primals_63, buf141, 144256, grid=grid(144256), stream=stream0)
        del convolution_49
        del primals_63
        del squeeze_94
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf142 = aten.convolution_backward(buf141, mul_225, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf141
        del mul_225
        del primals_156
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        buf145 = reinterpret_tensor(buf120, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf120  # reuse
        buf146 = reinterpret_tensor(buf145, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf145  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf146, buf143, relu_34, convolution_48, 2944, 49, grid=grid(2944), stream=stream0)
        buf147 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf146, buf147, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf148 = aten.convolution_backward(buf146, relu_35, primals_154, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf146
        del primals_154
        buf149 = buf148[0]
        buf150 = buf148[1]
        del buf148
        buf151 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf151, relu_35, 736, grid=grid(736), stream=stream0)
        del relu_35
        buf152 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf151, buf152, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf153 = aten.convolution_backward(buf151, mean_8, primals_152, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf151
        del mean_8
        del primals_152
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = empty((368, ), device='cuda', dtype=torch.float32)
        buf157 = empty((368, ), device='cuda', dtype=torch.float32)
        buf159 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_34, buf143, convolution_48, buf154, convolution_46, unsqueeze_334, squeeze_91, buf156, buf157, buf159, 368, 392, grid=grid(368), stream=stream0)
        buf158 = buf143; del buf143  # reuse
        buf160 = buf158; del buf158  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf160, relu_34, convolution_48, buf154, convolution_46, unsqueeze_334, buf157, squeeze_91, buf156, primals_61, 144256, grid=grid(144256), stream=stream0)
        del convolution_46
        del convolution_48
        del primals_61
        del relu_34
        del squeeze_91
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf161 = aten.convolution_backward(buf160, relu_33, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf160
        del primals_151
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf157; del buf157  # reuse
        buf165 = empty((368, ), device='cuda', dtype=torch.float32)
        buf166 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_33, buf162, convolution_45, unsqueeze_346, squeeze_88, buf164, buf165, buf166, 368, 392, grid=grid(368), stream=stream0)
        buf167 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf167, relu_33, convolution_45, unsqueeze_346, buf165, squeeze_88, buf164, primals_59, 144256, grid=grid(144256), stream=stream0)
        del convolution_45
        del primals_59
        del relu_33
        del squeeze_88
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf168 = aten.convolution_backward(buf167, relu_32, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_150
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf165; del buf165  # reuse
        buf172 = empty((368, ), device='cuda', dtype=torch.float32)
        buf174 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_16.run(relu_32, buf137, buf169, convolution_44, unsqueeze_358, squeeze_85, buf171, buf172, buf174, 368, 392, grid=grid(368), stream=stream0)
        buf173 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_17.run(relu_32, buf137, buf169, convolution_44, unsqueeze_358, buf172, squeeze_85, buf171, primals_57, buf173, 144256, grid=grid(144256), stream=stream0)
        del convolution_44
        del primals_57
        del squeeze_85
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf175 = aten.convolution_backward(buf173, mul_203, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf173
        del mul_203
        del primals_149
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = reinterpret_tensor(buf154, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf154  # reuse
        buf179 = reinterpret_tensor(buf178, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf178  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf179, buf176, relu_30, convolution_43, 2944, 49, grid=grid(2944), stream=stream0)
        buf180 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf179, buf180, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf181 = aten.convolution_backward(buf179, relu_31, primals_147, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf179
        del primals_147
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf184, relu_31, 736, grid=grid(736), stream=stream0)
        del relu_31
        buf185 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf184, buf185, 92, 8, grid=grid(92), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf186 = aten.convolution_backward(buf184, mean_7, primals_145, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf184
        del mean_7
        del primals_145
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = empty((368, ), device='cuda', dtype=torch.float32)
        buf190 = empty((368, ), device='cuda', dtype=torch.float32)
        buf192 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_30, buf176, convolution_43, buf187, convolution_41, unsqueeze_370, squeeze_82, buf189, buf190, buf192, 368, 392, grid=grid(368), stream=stream0)
        buf191 = buf176; del buf176  # reuse
        buf193 = buf191; del buf191  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf193, relu_30, convolution_43, buf187, convolution_41, unsqueeze_370, buf190, squeeze_82, buf189, primals_55, 144256, grid=grid(144256), stream=stream0)
        del convolution_41
        del convolution_43
        del primals_55
        del relu_30
        del squeeze_82
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf194 = aten.convolution_backward(buf193, relu_29, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf193
        del primals_144
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf197 = buf190; del buf190  # reuse
        buf198 = empty((368, ), device='cuda', dtype=torch.float32)
        buf199 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_29, buf195, convolution_40, unsqueeze_382, squeeze_79, buf197, buf198, buf199, 368, 392, grid=grid(368), stream=stream0)
        buf200 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf200, relu_29, convolution_40, unsqueeze_382, buf198, squeeze_79, buf197, primals_53, 144256, grid=grid(144256), stream=stream0)
        del convolution_40
        del primals_53
        del relu_29
        del squeeze_79
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf201 = aten.convolution_backward(buf200, relu_28, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf200
        del primals_143
        buf202 = buf201[0]
        buf203 = buf201[1]
        del buf201
        buf204 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_19.run(buf204, relu_28, relu_32, buf169, buf202, 144256, grid=grid(144256), stream=stream0)
        del relu_28
        del relu_32
        buf205 = buf198; del buf198  # reuse
        buf206 = empty((368, ), device='cuda', dtype=torch.float32)
        buf212 = empty((368, ), device='cuda', dtype=torch.float32)
        buf207 = empty((368, ), device='cuda', dtype=torch.float32)
        buf213 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf204, convolution_39, unsqueeze_394, convolution_38, unsqueeze_406, squeeze_76, squeeze_73, buf205, buf206, buf212, buf207, buf213, 368, 392, grid=grid(368), stream=stream0)
        buf208 = buf202; del buf202  # reuse
        buf214 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_21.run(buf204, convolution_39, unsqueeze_394, buf206, squeeze_76, buf205, primals_51, convolution_38, unsqueeze_406, buf212, squeeze_73, primals_49, buf208, buf214, 144256, grid=grid(144256), stream=stream0)
        del buf204
        del convolution_38
        del convolution_39
        del primals_49
        del primals_51
        del squeeze_73
        del squeeze_76
        del unsqueeze_394
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf209 = aten.convolution_backward(buf208, relu_24, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf208
        del primals_142
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf215 = aten.convolution_backward(buf214, mul_174, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf214
        del mul_174
        del primals_141
        buf216 = buf215[0]
        buf217 = buf215[1]
        del buf215
        buf218 = reinterpret_tensor(buf187, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf187  # reuse
        buf219 = reinterpret_tensor(buf218, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf218  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf219, buf216, relu_26, convolution_37, 2944, 49, grid=grid(2944), stream=stream0)
        buf220 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_4.run(buf219, buf220, 368, 8, grid=grid(368), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf221 = aten.convolution_backward(buf219, relu_27, primals_139, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf219
        del primals_139
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_22.run(buf224, relu_27, 304, grid=grid(304), stream=stream0)
        del relu_27
        buf225 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf224, buf225, 38, 8, grid=grid(38), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf226 = aten.convolution_backward(buf224, mean_6, primals_137, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf224
        del mean_6
        del primals_137
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf229 = buf206; del buf206  # reuse
        buf230 = empty((368, ), device='cuda', dtype=torch.float32)
        buf232 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(relu_26, buf216, convolution_37, buf227, convolution_35, unsqueeze_418, squeeze_70, buf229, buf230, buf232, 368, 392, grid=grid(368), stream=stream0)
        buf231 = buf216; del buf216  # reuse
        buf233 = buf231; del buf231  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf233, relu_26, convolution_37, buf227, convolution_35, unsqueeze_418, buf230, squeeze_70, buf229, primals_47, 144256, grid=grid(144256), stream=stream0)
        del buf227
        del convolution_35
        del convolution_37
        del primals_47
        del relu_26
        del squeeze_70
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf234 = aten.convolution_backward(buf233, relu_25, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
        del buf233
        del primals_136
        buf235 = buf234[0]
        buf236 = buf234[1]
        del buf234
        buf237 = buf230; del buf230  # reuse
        buf238 = empty((368, ), device='cuda', dtype=torch.float32)
        buf239 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(relu_25, buf235, convolution_34, unsqueeze_430, squeeze_67, buf237, buf238, buf239, 368, 1568, grid=grid(368), stream=stream0)
        buf240 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf240, relu_25, convolution_34, unsqueeze_430, buf238, squeeze_67, buf237, primals_45, 577024, grid=grid(577024), stream=stream0)
        del buf238
        del convolution_34
        del primals_45
        del relu_25
        del squeeze_67
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf241 = aten.convolution_backward(buf240, relu_24, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf240
        del primals_135
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf244 = empty((152, ), device='cuda', dtype=torch.float32)
        buf245 = empty((152, ), device='cuda', dtype=torch.float32)
        buf247 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_24, buf210, buf242, convolution_33, unsqueeze_442, squeeze_64, buf244, buf245, buf247, 152, 1568, grid=grid(152), stream=stream0)
        buf246 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_27.run(relu_24, buf210, buf242, convolution_33, unsqueeze_442, buf245, squeeze_64, buf244, primals_43, buf246, 238336, grid=grid(238336), stream=stream0)
        del convolution_33
        del primals_43
        del squeeze_64
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf248 = aten.convolution_backward(buf246, mul_152, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf246
        del mul_152
        del primals_134
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf252 = reinterpret_tensor(buf251, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf251  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28.run(buf252, buf249, relu_22, convolution_32, 1216, 196, grid=grid(1216), stream=stream0)
        buf253 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf252, buf253, 152, 8, grid=grid(152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf254 = aten.convolution_backward(buf252, relu_23, primals_132, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf252
        del primals_132
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_22.run(buf257, relu_23, 304, grid=grid(304), stream=stream0)
        del relu_23
        buf258 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf257, buf258, 38, 8, grid=grid(38), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf259 = aten.convolution_backward(buf257, mean_5, primals_130, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf257
        del mean_5
        del primals_130
        buf260 = buf259[0]
        buf261 = buf259[1]
        del buf259
        buf262 = empty((152, ), device='cuda', dtype=torch.float32)
        buf263 = empty((152, ), device='cuda', dtype=torch.float32)
        buf265 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30.run(relu_22, buf249, convolution_32, buf260, convolution_30, unsqueeze_454, squeeze_61, buf262, buf263, buf265, 152, 1568, grid=grid(152), stream=stream0)
        buf264 = buf249; del buf249  # reuse
        buf266 = buf264; del buf264  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31.run(buf266, relu_22, convolution_32, buf260, convolution_30, unsqueeze_454, buf263, squeeze_61, buf262, primals_41, 238336, grid=grid(238336), stream=stream0)
        del convolution_30
        del convolution_32
        del primals_41
        del relu_22
        del squeeze_61
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf267 = aten.convolution_backward(buf266, relu_21, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
        del buf266
        del primals_129
        buf268 = buf267[0]
        buf269 = buf267[1]
        del buf267
        buf270 = buf263; del buf263  # reuse
        buf271 = empty((152, ), device='cuda', dtype=torch.float32)
        buf272 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_32.run(relu_21, buf268, convolution_29, unsqueeze_466, squeeze_58, buf270, buf271, buf272, 152, 1568, grid=grid(152), stream=stream0)
        buf273 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33.run(buf273, relu_21, convolution_29, unsqueeze_466, buf271, squeeze_58, buf270, primals_39, 238336, grid=grid(238336), stream=stream0)
        del convolution_29
        del primals_39
        del relu_21
        del squeeze_58
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf274 = aten.convolution_backward(buf273, relu_20, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf273
        del primals_128
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_34.run(buf277, relu_20, relu_24, buf242, buf275, 238336, grid=grid(238336), stream=stream0)
        del buf242
        del relu_20
        del relu_24
        buf278 = buf271; del buf271  # reuse
        buf279 = empty((152, ), device='cuda', dtype=torch.float32)
        buf280 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_35.run(buf277, convolution_28, unsqueeze_478, squeeze_55, buf278, buf279, buf280, 152, 1568, grid=grid(152), stream=stream0)
        buf281 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_36.run(buf277, convolution_28, unsqueeze_478, buf279, squeeze_55, buf278, primals_37, buf281, 238336, grid=grid(238336), stream=stream0)
        del convolution_28
        del primals_37
        del squeeze_55
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf282 = aten.convolution_backward(buf281, mul_130, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf281
        del mul_130
        del primals_127
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        buf285 = reinterpret_tensor(buf260, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf260  # reuse
        buf286 = reinterpret_tensor(buf285, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf285  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28.run(buf286, buf283, relu_18, convolution_27, 1216, 196, grid=grid(1216), stream=stream0)
        buf287 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf286, buf287, 152, 8, grid=grid(152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf288 = aten.convolution_backward(buf286, relu_19, primals_125, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf286
        del primals_125
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf291 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_22.run(buf291, relu_19, 304, grid=grid(304), stream=stream0)
        del relu_19
        buf292 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf291, buf292, 38, 8, grid=grid(38), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf293 = aten.convolution_backward(buf291, mean_4, primals_123, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf291
        del mean_4
        del primals_123
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf296 = empty((152, ), device='cuda', dtype=torch.float32)
        buf297 = empty((152, ), device='cuda', dtype=torch.float32)
        buf299 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30.run(relu_18, buf283, convolution_27, buf294, convolution_25, unsqueeze_490, squeeze_52, buf296, buf297, buf299, 152, 1568, grid=grid(152), stream=stream0)
        buf298 = buf283; del buf283  # reuse
        buf300 = buf298; del buf298  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31.run(buf300, relu_18, convolution_27, buf294, convolution_25, unsqueeze_490, buf297, squeeze_52, buf296, primals_35, 238336, grid=grid(238336), stream=stream0)
        del convolution_25
        del convolution_27
        del primals_35
        del relu_18
        del squeeze_52
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf301 = aten.convolution_backward(buf300, relu_17, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
        del buf300
        del primals_122
        buf302 = buf301[0]
        buf303 = buf301[1]
        del buf301
        buf304 = buf297; del buf297  # reuse
        buf305 = empty((152, ), device='cuda', dtype=torch.float32)
        buf306 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_32.run(relu_17, buf302, convolution_24, unsqueeze_502, squeeze_49, buf304, buf305, buf306, 152, 1568, grid=grid(152), stream=stream0)
        buf307 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33.run(buf307, relu_17, convolution_24, unsqueeze_502, buf305, squeeze_49, buf304, primals_33, 238336, grid=grid(238336), stream=stream0)
        del convolution_24
        del primals_33
        del relu_17
        del squeeze_49
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf308 = aten.convolution_backward(buf307, relu_16, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_121
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf311 = buf305; del buf305  # reuse
        buf312 = empty((152, ), device='cuda', dtype=torch.float32)
        buf314 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_16, buf277, buf309, convolution_23, unsqueeze_514, squeeze_46, buf311, buf312, buf314, 152, 1568, grid=grid(152), stream=stream0)
        buf313 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_27.run(relu_16, buf277, buf309, convolution_23, unsqueeze_514, buf312, squeeze_46, buf311, primals_31, buf313, 238336, grid=grid(238336), stream=stream0)
        del convolution_23
        del primals_31
        del squeeze_46
        del unsqueeze_514
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf315 = aten.convolution_backward(buf313, mul_108, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf313
        del mul_108
        del primals_120
        buf316 = buf315[0]
        buf317 = buf315[1]
        del buf315
        buf318 = reinterpret_tensor(buf294, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf294  # reuse
        buf319 = reinterpret_tensor(buf318, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf318  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28.run(buf319, buf316, relu_14, convolution_22, 1216, 196, grid=grid(1216), stream=stream0)
        buf320 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf319, buf320, 152, 8, grid=grid(152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf321 = aten.convolution_backward(buf319, relu_15, primals_118, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf319
        del primals_118
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf324 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_22.run(buf324, relu_15, 304, grid=grid(304), stream=stream0)
        del relu_15
        buf325 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf324, buf325, 38, 8, grid=grid(38), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf326 = aten.convolution_backward(buf324, mean_3, primals_116, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf324
        del mean_3
        del primals_116
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = empty((152, ), device='cuda', dtype=torch.float32)
        buf330 = empty((152, ), device='cuda', dtype=torch.float32)
        buf332 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30.run(relu_14, buf316, convolution_22, buf327, convolution_20, unsqueeze_526, squeeze_43, buf329, buf330, buf332, 152, 1568, grid=grid(152), stream=stream0)
        buf331 = buf316; del buf316  # reuse
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31.run(buf333, relu_14, convolution_22, buf327, convolution_20, unsqueeze_526, buf330, squeeze_43, buf329, primals_29, 238336, grid=grid(238336), stream=stream0)
        del convolution_20
        del convolution_22
        del primals_29
        del relu_14
        del squeeze_43
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf334 = aten.convolution_backward(buf333, relu_13, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
        del buf333
        del primals_115
        buf335 = buf334[0]
        buf336 = buf334[1]
        del buf334
        buf337 = buf330; del buf330  # reuse
        buf338 = empty((152, ), device='cuda', dtype=torch.float32)
        buf339 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_32.run(relu_13, buf335, convolution_19, unsqueeze_538, squeeze_40, buf337, buf338, buf339, 152, 1568, grid=grid(152), stream=stream0)
        buf340 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33.run(buf340, relu_13, convolution_19, unsqueeze_538, buf338, squeeze_40, buf337, primals_27, 238336, grid=grid(238336), stream=stream0)
        del convolution_19
        del primals_27
        del relu_13
        del squeeze_40
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf341 = aten.convolution_backward(buf340, relu_12, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf340
        del primals_114
        buf342 = buf341[0]
        buf343 = buf341[1]
        del buf341
        buf344 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_34.run(buf344, relu_12, relu_16, buf309, buf342, 238336, grid=grid(238336), stream=stream0)
        del relu_12
        del relu_16
        buf345 = buf338; del buf338  # reuse
        buf346 = empty((152, ), device='cuda', dtype=torch.float32)
        buf352 = empty((152, ), device='cuda', dtype=torch.float32)
        buf347 = empty((152, ), device='cuda', dtype=torch.float32)
        buf353 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf344, convolution_18, unsqueeze_550, convolution_17, unsqueeze_562, squeeze_37, squeeze_34, buf345, buf346, buf352, buf347, buf353, 152, 1568, grid=grid(152), stream=stream0)
        buf348 = buf342; del buf342  # reuse
        buf354 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_38.run(buf344, convolution_18, unsqueeze_550, buf346, squeeze_37, buf345, primals_25, convolution_17, unsqueeze_562, buf352, squeeze_34, primals_23, buf348, buf354, 238336, grid=grid(238336), stream=stream0)
        del buf344
        del convolution_17
        del convolution_18
        del primals_23
        del primals_25
        del squeeze_34
        del squeeze_37
        del unsqueeze_550
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf349 = aten.convolution_backward(buf348, relu_8, primals_113, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf348
        del primals_113
        buf350 = buf349[0]
        buf351 = buf349[1]
        del buf349
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf355 = aten.convolution_backward(buf354, mul_79, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf354
        del mul_79
        del primals_112
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        buf358 = reinterpret_tensor(buf327, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf327  # reuse
        buf359 = reinterpret_tensor(buf358, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf358  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_28.run(buf359, buf356, relu_10, convolution_16, 1216, 196, grid=grid(1216), stream=stream0)
        buf360 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf359, buf360, 152, 8, grid=grid(152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf361 = aten.convolution_backward(buf359, relu_11, primals_110, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf359
        del primals_110
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_39.run(buf364, relu_11, 112, grid=grid(112), stream=stream0)
        del relu_11
        buf365 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_40.run(buf364, buf365, 14, 8, grid=grid(14), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf366 = aten.convolution_backward(buf364, mean_2, primals_108, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf364
        del mean_2
        del primals_108
        buf367 = buf366[0]
        buf368 = buf366[1]
        del buf366
        buf369 = buf346; del buf346  # reuse
        buf370 = empty((152, ), device='cuda', dtype=torch.float32)
        buf372 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30.run(relu_10, buf356, convolution_16, buf367, convolution_14, unsqueeze_574, squeeze_31, buf369, buf370, buf372, 152, 1568, grid=grid(152), stream=stream0)
        buf371 = buf356; del buf356  # reuse
        buf373 = buf371; del buf371  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_31.run(buf373, relu_10, convolution_16, buf367, convolution_14, unsqueeze_574, buf370, squeeze_31, buf369, primals_21, 238336, grid=grid(238336), stream=stream0)
        del buf367
        del convolution_14
        del convolution_16
        del primals_21
        del relu_10
        del squeeze_31
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf374 = aten.convolution_backward(buf373, relu_9, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
        del buf373
        del primals_107
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf377 = buf370; del buf370  # reuse
        buf378 = empty((152, ), device='cuda', dtype=torch.float32)
        buf379 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(relu_9, buf375, convolution_13, unsqueeze_586, squeeze_28, buf377, buf378, buf379, 152, 6272, grid=grid(152), stream=stream0)
        buf380 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42.run(buf380, relu_9, convolution_13, unsqueeze_586, buf378, squeeze_28, buf377, primals_19, 953344, grid=grid(953344), stream=stream0)
        del buf378
        del convolution_13
        del primals_19
        del relu_9
        del squeeze_28
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf381 = aten.convolution_backward(buf380, relu_8, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf380
        del primals_106
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = empty((56, ), device='cuda', dtype=torch.float32)
        buf385 = empty((56, ), device='cuda', dtype=torch.float32)
        buf391 = empty((56, ), device='cuda', dtype=torch.float32)
        buf387 = empty((56, ), device='cuda', dtype=torch.float32)
        buf393 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu_8, buf350, buf382, convolution_12, unsqueeze_598, convolution_11, unsqueeze_610, squeeze_25, squeeze_22, buf384, buf385, buf391, buf387, buf393, 56, 6272, grid=grid(56), stream=stream0)
        buf386 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        buf392 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44.run(relu_8, buf350, buf382, convolution_12, unsqueeze_598, buf385, squeeze_25, buf384, primals_17, convolution_11, unsqueeze_610, buf391, squeeze_22, primals_15, buf386, buf392, 351232, grid=grid(351232), stream=stream0)
        del buf350
        del buf382
        del convolution_11
        del convolution_12
        del primals_15
        del primals_17
        del relu_8
        del squeeze_22
        del squeeze_25
        del unsqueeze_598
        del unsqueeze_610
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf388 = aten.convolution_backward(buf386, relu_4, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf386
        del primals_105
        buf389 = buf388[0]
        buf390 = buf388[1]
        del buf388
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf394 = aten.convolution_backward(buf392, mul_50, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf392
        del mul_50
        del primals_104
        buf395 = buf394[0]
        buf396 = buf394[1]
        del buf394
        buf397 = empty_strided((8, 56, 1, 1), (56, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf398 = reinterpret_tensor(buf397, (8, 56, 1, 1), (56, 1, 1, 1), 0); del buf397  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_45.run(buf398, buf395, relu_6, convolution_10, 448, 784, grid=grid(448), stream=stream0)
        buf399 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_46.run(buf398, buf399, 56, 8, grid=grid(56), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf400 = aten.convolution_backward(buf398, relu_7, primals_102, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf398
        del primals_102
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_47.run(buf403, relu_7, 48, grid=grid(48), stream=stream0)
        del relu_7
        buf404 = empty((6, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_48.run(buf403, buf404, 6, 8, grid=grid(6), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf405 = aten.convolution_backward(buf403, mean_1, primals_100, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf403
        del mean_1
        del primals_100
        buf406 = buf405[0]
        buf407 = buf405[1]
        del buf405
        buf408 = buf385; del buf385  # reuse
        buf409 = empty((56, ), device='cuda', dtype=torch.float32)
        buf411 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_49.run(relu_6, buf395, convolution_10, buf406, convolution_8, unsqueeze_622, squeeze_19, buf408, buf409, buf411, 56, 6272, grid=grid(56), stream=stream0)
        buf410 = buf395; del buf395  # reuse
        buf412 = buf410; del buf410  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_50.run(buf412, relu_6, convolution_10, buf406, convolution_8, unsqueeze_622, buf409, squeeze_19, buf408, primals_13, 351232, grid=grid(351232), stream=stream0)
        del buf406
        del convolution_10
        del convolution_8
        del primals_13
        del relu_6
        del squeeze_19
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf413 = aten.convolution_backward(buf412, relu_5, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 7, [True, True, False])
        del buf412
        del primals_99
        buf414 = buf413[0]
        buf415 = buf413[1]
        del buf413
        buf416 = empty_strided((56, 4), (1, 56), device='cuda', dtype=torch.float32)
        buf418 = empty_strided((56, 4), (1, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(relu_5, buf414, convolution_7, unsqueeze_634, buf416, buf418, 224, 6272, grid=grid(224), stream=stream0)
        buf417 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_52.run(buf416, buf417, 56, 4, grid=grid(56), stream=stream0)
        del buf416
        buf419 = empty((56, ), device='cuda', dtype=torch.float32)
        buf420 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_53.run(buf418, squeeze_16, buf419, buf420, 56, 4, grid=grid(56), stream=stream0)
        del buf418
        buf421 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54.run(buf421, relu_5, convolution_7, unsqueeze_634, buf419, squeeze_16, buf417, primals_11, 1404928, grid=grid(1404928), stream=stream0)
        del buf419
        del convolution_7
        del primals_11
        del relu_5
        del squeeze_16
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf422 = aten.convolution_backward(buf421, relu_4, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf421
        del primals_98
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf425 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        buf427 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(relu_4, buf389, buf423, convolution_6, unsqueeze_646, convolution_5, unsqueeze_658, buf425, buf427, buf434, 96, 6272, grid=grid(96), stream=stream0)
        buf426 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_56.run(buf425, buf426, 24, 4, grid=grid(24), stream=stream0)
        del buf425
        buf428 = empty((24, ), device='cuda', dtype=torch.float32)
        buf430 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_57.run(buf427, squeeze_13, buf428, buf430, 24, 4, grid=grid(24), stream=stream0)
        buf435 = empty((24, ), device='cuda', dtype=torch.float32)
        buf437 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_57.run(buf434, squeeze_10, buf435, buf437, 24, 4, grid=grid(24), stream=stream0)
        buf429 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        buf436 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58.run(relu_4, buf389, buf423, convolution_6, unsqueeze_646, buf428, squeeze_13, buf426, primals_9, convolution_5, unsqueeze_658, buf435, squeeze_10, primals_7, buf429, buf436, 602112, grid=grid(602112), stream=stream0)
        del buf389
        del buf423
        del convolution_5
        del convolution_6
        del primals_7
        del primals_9
        del relu_4
        del squeeze_10
        del squeeze_13
        del unsqueeze_646
        del unsqueeze_658
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf431 = aten.convolution_backward(buf429, relu, primals_97, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf429
        del primals_97
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf438 = aten.convolution_backward(buf436, mul_21, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf436
        del mul_21
        del primals_96
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = empty_strided((8, 24, 1, 1), (24, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf442 = reinterpret_tensor(buf441, (8, 24, 1, 1), (24, 1, 1, 1), 0); del buf441  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_mul_sigmoid_sigmoid_backward_sum_59.run(buf442, buf439, relu_2, convolution_4, 192, 3136, grid=grid(192), stream=stream0)
        buf443 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf442, buf443, 24, 8, grid=grid(24), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf444 = aten.convolution_backward(buf442, relu_3, primals_94, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf442
        del primals_94
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf447 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_61.run(buf447, relu_3, 64, grid=grid(64), stream=stream0)
        del relu_3
        buf448 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_62.run(buf447, buf448, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf449 = aten.convolution_backward(buf447, mean, primals_92, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf447
        del mean
        del primals_92
        buf450 = buf449[0]
        buf451 = buf449[1]
        del buf449
        buf452 = buf434; del buf434  # reuse
        buf454 = buf427; del buf427  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_63.run(relu_2, buf439, convolution_4, buf450, convolution_2, unsqueeze_670, buf452, buf454, 96, 6272, grid=grid(96), stream=stream0)
        buf453 = buf428; del buf428  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_56.run(buf452, buf453, 24, 4, grid=grid(24), stream=stream0)
        del buf452
        buf455 = empty((24, ), device='cuda', dtype=torch.float32)
        buf457 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_57.run(buf454, squeeze_7, buf455, buf457, 24, 4, grid=grid(24), stream=stream0)
        del buf454
        buf456 = buf439; del buf439  # reuse
        buf458 = buf456; del buf456  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_64.run(buf458, relu_2, convolution_4, buf450, convolution_2, unsqueeze_670, buf455, squeeze_7, buf453, primals_5, 602112, grid=grid(602112), stream=stream0)
        del buf450
        del convolution_2
        del convolution_4
        del primals_5
        del relu_2
        del squeeze_7
        del unsqueeze_670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf459 = aten.convolution_backward(buf458, relu_1, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 3, [True, True, False])
        del buf458
        del primals_91
        buf460 = buf459[0]
        buf461 = buf459[1]
        del buf459
        buf462 = empty((24, 13), device='cuda', dtype=torch.float32)
        buf464 = empty((24, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(relu_1, buf460, convolution_1, unsqueeze_682, buf462, buf464, 312, 7720, grid=grid(312), stream=stream0)
        buf463 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_66.run(buf462, buf463, 24, 13, grid=grid(24), stream=stream0)
        del buf462
        buf465 = empty((24, ), device='cuda', dtype=torch.float32)
        buf466 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf464, squeeze_4, buf465, buf466, 24, 13, grid=grid(24), stream=stream0)
        del buf464
        buf467 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68.run(buf467, relu_1, convolution_1, unsqueeze_682, buf465, squeeze_4, buf463, primals_3, 2408448, grid=grid(2408448), stream=stream0)
        del buf465
        del convolution_1
        del primals_3
        del relu_1
        del squeeze_4
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf468 = aten.convolution_backward(buf467, relu, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf467
        del primals_90
        buf469 = buf468[0]
        buf470 = buf468[1]
        del buf468
        buf471 = empty((32, 13), device='cuda', dtype=torch.float32)
        buf473 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_69.run(relu, buf432, buf469, convolution, unsqueeze_694, buf471, buf473, 416, 7720, grid=grid(416), stream=stream0)
        buf472 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_70.run(buf471, buf472, 32, 13, grid=grid(32), stream=stream0)
        del buf471
        buf474 = empty((32, ), device='cuda', dtype=torch.float32)
        buf476 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_71.run(buf473, squeeze_1, buf474, buf476, 32, 13, grid=grid(32), stream=stream0)
        del buf473
        buf475 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_72.run(buf475, relu, buf469, convolution, unsqueeze_694, buf474, squeeze_1, buf472, primals_1, 3211264, grid=grid(3211264), stream=stream0)
        del buf469
        del buf474
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf477 = aten.convolution_backward(buf475, primals_319, primals_89, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf475
        del primals_319
        del primals_89
        buf478 = buf477[1]
        return (buf476, buf472, buf466, buf463, buf457, buf453, buf437, buf426, buf430, buf426, buf420, buf417, buf411, buf408, buf393, buf384, buf387, buf384, buf379, buf377, buf372, buf369, buf353, buf345, buf347, buf345, buf339, buf337, buf332, buf329, buf314, buf311, buf306, buf304, buf299, buf296, buf280, buf278, buf272, buf270, buf265, buf262, buf247, buf244, buf239, buf237, buf232, buf229, buf213, buf205, buf207, buf205, buf199, buf197, buf192, buf189, buf174, buf171, buf166, buf164, buf159, buf156, buf140, buf138, buf132, buf130, buf125, buf122, buf107, buf104, buf99, buf97, buf92, buf89, buf73, buf71, buf65, buf63, buf58, buf55, buf39, buf36, buf31, buf29, buf24, buf21, buf5, buf3, buf478, buf470, buf461, buf451, buf448, buf446, buf443, buf440, buf433, buf424, buf415, buf407, buf404, buf402, buf399, buf396, buf390, buf383, buf376, buf368, buf365, buf363, buf360, buf357, buf351, buf343, buf336, buf328, buf325, buf323, buf320, buf317, buf310, buf303, buf295, buf292, buf290, buf287, buf284, buf276, buf269, buf261, buf258, buf256, buf253, buf250, buf243, buf236, buf228, buf225, buf223, buf220, buf217, buf211, buf203, buf196, buf188, buf185, buf183, buf180, buf177, buf170, buf163, buf155, buf152, buf150, buf147, buf144, buf136, buf129, buf121, buf118, buf116, buf113, buf110, buf103, buf96, buf88, buf85, buf83, buf80, buf77, buf69, buf62, buf54, buf51, buf49, buf46, buf43, buf35, buf28, buf20, buf17, buf15, buf12, buf9, reinterpret_tensor(buf1, (1000, 368), (368, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((24, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((56, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 24, 112, 112), (301056, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 24, 112, 112), (301056, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 24, 56, 56), (75264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 56, 56, 56), (175616, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 56, 56, 56), (175616, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 56, 28, 28), (43904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 152, 28, 28), (119168, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 152, 28, 28), (119168, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_130 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 368, 14, 14), (72128, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 368, 14, 14), (72128, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_203 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_225 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_247 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_269 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_291 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 368), (368, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 368), (368, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 368, 7, 7), (18032, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_178 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, squeeze_19, relu_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, convolution_12, squeeze_25, relu_8, convolution_13, squeeze_28, relu_9, convolution_14, squeeze_31, relu_10, mean_2, relu_11, convolution_16, mul_79, convolution_17, squeeze_34, convolution_18, squeeze_37, relu_12, convolution_19, squeeze_40, relu_13, convolution_20, squeeze_43, relu_14, mean_3, relu_15, convolution_22, mul_108, convolution_23, squeeze_46, relu_16, convolution_24, squeeze_49, relu_17, convolution_25, squeeze_52, relu_18, mean_4, relu_19, convolution_27, mul_130, convolution_28, squeeze_55, relu_20, convolution_29, squeeze_58, relu_21, convolution_30, squeeze_61, relu_22, mean_5, relu_23, convolution_32, mul_152, convolution_33, squeeze_64, relu_24, convolution_34, squeeze_67, relu_25, convolution_35, squeeze_70, relu_26, mean_6, relu_27, convolution_37, mul_174, convolution_38, squeeze_73, convolution_39, squeeze_76, relu_28, convolution_40, squeeze_79, relu_29, convolution_41, squeeze_82, relu_30, mean_7, relu_31, convolution_43, mul_203, convolution_44, squeeze_85, relu_32, convolution_45, squeeze_88, relu_33, convolution_46, squeeze_91, relu_34, mean_8, relu_35, convolution_48, mul_225, convolution_49, squeeze_94, relu_36, convolution_50, squeeze_97, relu_37, convolution_51, squeeze_100, relu_38, mean_9, relu_39, convolution_53, mul_247, convolution_54, squeeze_103, relu_40, convolution_55, squeeze_106, relu_41, convolution_56, squeeze_109, relu_42, mean_10, relu_43, convolution_58, mul_269, convolution_59, squeeze_112, relu_44, convolution_60, squeeze_115, relu_45, convolution_61, squeeze_118, relu_46, mean_11, relu_47, convolution_63, mul_291, convolution_64, squeeze_121, relu_48, convolution_65, squeeze_124, relu_49, convolution_66, squeeze_127, relu_50, mean_12, relu_51, convolution_68, mul_313, convolution_69, squeeze_130, clone, permute_1, le, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
