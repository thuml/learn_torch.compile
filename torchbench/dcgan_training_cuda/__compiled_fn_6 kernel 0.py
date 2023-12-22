
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


# kernel path: /tmp/torchinductor_youkaichao/cl/cclicsawo22gj2khlmatxekdoggaf4r7ufez2hgdgv3yr4prfqtu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.sigmoid_backward]

triton_poi_fused_convolution_backward_sigmoid_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_sigmoid_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/tr/ctr2yls6eqwrklc2z66dgsdsnbcs2yaias7hpz22x5ytdjsmqf7k.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = (rindex // 16)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceaohqwonortedzlv6m6u3mvk77367heehp3wyyocfgamtcd4ntt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yawcmmfn5emo4mejg7kt3n66jym7ubnkilcpfxnbmmefxadcy4.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3ocewy4mi2azsxc3neztnn5xjgmpa2qorx6nqrxn2z5rwvxcjfp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvg6pje3lwp2v5r3kgxj6rmgm5rhyqk6wuxvliybzo2xiwp7jus4.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 256
    r2 = (rindex // 256)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwmei75fz7nuobwrxzen7xpj4ptgc3jgt5gzzz5m42ke6ozt66v.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpwowagu3pd7tinavqjqkye6l3gc5awrr76tzzclvnzmfrr4voi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, where, convolution_1, where_1, convolution_2, where_2, convolution_3, where_3, sigmoid, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_8, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_11, (1, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_21, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(where, (4, 64, 32, 32), (65536, 1024, 32, 1))
    assert_size_stride(convolution_1, (4, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(where_1, (4, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(convolution_2, (4, 256, 8, 8), (16384, 64, 8, 1))
    assert_size_stride(where_2, (4, 256, 8, 8), (16384, 64, 8, 1))
    assert_size_stride(convolution_3, (4, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(where_3, (4, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(sigmoid, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(tangents_1, (4, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.sigmoid_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_backward_sigmoid_backward_0.run(tangents_1, sigmoid, buf0, 4, grid=grid(4), stream=stream0)
        del sigmoid
        del tangents_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.sigmoid_backward]
        buf1 = aten.convolution_backward(buf0, where_3, primals_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf0
        del primals_11
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        buf4 = empty((512, ), device='cuda', dtype=torch.float32)
        buf5 = empty((512, ), device='cuda', dtype=torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_1.run(buf6, where_3, buf2, convolution_3, primals_18, primals_19, buf4, 512, 64, grid=grid(512), stream=stream0)
        del convolution_3
        del primals_18
        buf7 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2.run(buf7, where_3, primals_19, primals_9, 32768, grid=grid(32768), stream=stream0)
        del primals_19
        del primals_9
        del where_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf8 = aten.convolution_backward(buf7, where_2, primals_8, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf7
        del primals_8
        buf9 = buf8[0]
        buf10 = buf8[1]
        del buf8
        buf11 = empty((256, ), device='cuda', dtype=torch.float32)
        buf12 = empty((256, ), device='cuda', dtype=torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_3.run(buf13, where_2, buf9, convolution_2, primals_15, primals_16, buf11, 256, 256, grid=grid(256), stream=stream0)
        del convolution_2
        del primals_15
        buf14 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_4.run(buf14, where_2, primals_16, primals_6, 65536, grid=grid(65536), stream=stream0)
        del primals_16
        del primals_6
        del where_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf15 = aten.convolution_backward(buf14, where_1, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf14
        del primals_5
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty((128, ), device='cuda', dtype=torch.float32)
        buf19 = empty((128, ), device='cuda', dtype=torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_5.run(buf20, where_1, buf16, convolution_1, primals_12, primals_13, buf18, 128, 1024, grid=grid(128), stream=stream0)
        del convolution_1
        del primals_12
        buf21 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6.run(buf21, where_1, primals_13, primals_3, 131072, grid=grid(131072), stream=stream0)
        del primals_13
        del primals_3
        del where_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf22 = aten.convolution_backward(buf21, where, primals_2, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf21
        del primals_2
        buf23 = buf22[0]
        buf24 = buf22[1]
        del buf22
        buf25 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_7.run(buf25, where, 262144, grid=grid(262144), stream=stream0)
        del where
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward]
        buf26 = aten.convolution_backward(buf25, primals_21, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf25
        del primals_1
        del primals_21
        buf27 = buf26[1]
        return (buf27, buf24, buf20, buf18, buf17, buf13, buf11, buf10, buf6, buf4, buf3, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    where = rand_strided((4, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    where_1 = rand_strided((4, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 256, 8, 8), (16384, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    where_2 = rand_strided((4, 256, 8, 8), (16384, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    where_3 = rand_strided((4, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    sigmoid = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, where, convolution_1, where_1, convolution_2, where_2, convolution_3, where_3, sigmoid, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dcgan', benchmark_compiled_module)
