
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chgeggahhxih666znmrl7x3yvqgljdl46gwyr4jhyv5x4mogow6k.py
# Source Nodes: [key_states], Original ATen: [aten.clone]
# key_states => clone
triton_poi_fused_clone_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/7n/c7ns5vtt766nrwoavtfi5fkh6r66cgha7ascaef55bl3amy264pp.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_2
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdolblwwsworflzrnclje52rtrjaexiwwoovj76jmmnjk2zvyviz.py
# Source Nodes: [attn_probs, attn_weights_3], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_probs => clone_3
# attn_weights_3 => amax, div, exp, sub, sum_1
triton_per_fused__softmax_clone_detach_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6xjci7epm3revfpgn4wo35fluyfll7c5r2u3vfc3zawvtrnyxs.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c467wb33uluobshsevvsqk6ppzly5sdsb3yid3hsymcqprennbmq.py
# Source Nodes: [hidden_states_2, l__self___fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_2 => add_1
# l__self___fc1 => view_18
# residual_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 256.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7o2gcf52idtzgcye6jbp4bq3akgjrcwbfikpb6vzclv433zxkq6.py
# Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
# hidden_states_4 => relu
# hidden_states_6 => view_20
triton_poi_fused_relu_threshold_backward_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpatkfwqsgbulwm3vcdvuxwoyhfaj2biuyi45vubgpyr4bn7en6.py
# Source Nodes: [hidden_states_8, hidden_states_9, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_8 => add_4
# hidden_states_9 => add_5, add_6, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
# residual_1 => add_3, mul_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 256.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
    args.clear()
    assert_size_stride(primals_1, (256, 256), (256, 1))
    assert_size_stride(primals_2, (256, ), (1, ))
    assert_size_stride(primals_3, (256, 256), (256, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, 256), (256, 1))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (256, 256), (256, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (2048, 256), (256, 1))
    assert_size_stride(primals_12, (2048, ), (1, ))
    assert_size_stride(primals_13, (256, 2048), (2048, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(primals_18, (1, 1, 128, 128), (16384, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_17, (128, 256), (256, 1), 0), reinterpret_tensor(primals_1, (256, 256), (1, 256), 0), out=buf0)
        buf1 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_17, (128, 256), (256, 1), 0), reinterpret_tensor(primals_3, (256, 256), (1, 256), 0), out=buf1)
        buf2 = empty((1, 4, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_clone_0.run(buf1, primals_4, buf2, 32768, grid=grid(32768), stream=stream0)
        del primals_4
        buf3 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_17, (128, 256), (256, 1), 0), reinterpret_tensor(primals_5, (256, 256), (1, 256), 0), out=buf3)
        buf4 = empty((1, 4, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf3, primals_6, buf4, 32768, grid=grid(32768), stream=stream0)
        del primals_6
        buf5 = reinterpret_tensor(buf3, (1, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf0, primals_2, buf5, 32768, grid=grid(32768), stream=stream0)
        del primals_2
        buf6 = empty((4, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf2, (4, 64, 128), (8192, 1, 64), 0), out=buf6)
        buf9 = empty((4, 128, 128), device='cuda', dtype=torch.float32)
        buf35 = empty((4, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs, attn_weights_3], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_2.run(buf6, primals_18, buf9, buf35, 512, 128, grid=grid(512), stream=stream0)
        del buf6
        del primals_18
        buf10 = reinterpret_tensor(buf0, (4, 128, 64), (8192, 64, 1), 0); del buf0  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf9, reinterpret_tensor(buf4, (4, 128, 64), (8192, 64, 1), 0), out=buf10)
        buf11 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf10, buf11, 32768, grid=grid(32768), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (128, 256), (256, 1), 0); del buf10  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf11, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf12)
        del primals_8
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf13 = aten.native_dropout(reinterpret_tensor(buf12, (1, 128, 256), (32768, 256, 1), 0), 0.1, True)
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf19 = reinterpret_tensor(buf12, (1, 128, 256), (32768, 256, 1), 0); del buf12  # reuse
        buf20 = empty((128, 256), device='cuda', dtype=torch.float32)
        buf34 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_2, l__self___fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(primals_17, buf14, primals_9, primals_10, buf19, buf20, buf34, 128, 256, grid=grid(128), stream=stream0)
        buf21 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_11, (256, 2048), (1, 256), 0), out=buf21)
        buf22 = empty((128, 2048), device='cuda', dtype=torch.float32)
        buf33 = empty((1, 128, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_5.run(buf21, primals_12, buf22, buf33, 262144, grid=grid(262144), stream=stream0)
        del buf21
        del primals_12
        buf23 = reinterpret_tensor(buf14, (128, 256), (256, 1), 0); del buf14  # reuse
        # Source Nodes: [hidden_states_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf22, reinterpret_tensor(primals_13, (2048, 256), (1, 2048), 0), alpha=1, beta=1, out=buf23)
        del primals_14
        # Source Nodes: [hidden_states_7], Original ATen: [aten.native_dropout]
        buf24 = aten.native_dropout(reinterpret_tensor(buf23, (1, 128, 256), (32768, 256, 1), 0), 0.1, True)
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf30 = reinterpret_tensor(buf23, (1, 128, 256), (32768, 256, 1), 0); del buf23  # reuse
        buf31 = empty((1, 128, 256), device='cuda', dtype=torch.float32)
        buf32 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, hidden_states_9, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6.run(buf19, primals_9, primals_10, buf25, primals_15, primals_16, buf30, buf31, buf32, 128, 256, grid=grid(128), stream=stream0)
        del buf25
        del primals_10
        del primals_16
        return (buf31, buf2, buf4, primals_9, primals_15, reinterpret_tensor(primals_17, (128, 256), (256, 1), 0), buf11, buf15, buf19, buf20, buf22, buf26, buf30, buf32, reinterpret_tensor(primals_13, (256, 2048), (2048, 1), 0), buf33, reinterpret_tensor(primals_11, (2048, 256), (256, 1), 0), buf34, reinterpret_tensor(primals_7, (256, 256), (256, 1), 0), reinterpret_tensor(buf9, (4, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf4, (4, 64, 128), (8192, 1, 64), 0), buf35, reinterpret_tensor(buf5, (4, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf2, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_5, (256, 256), (256, 1), 0), reinterpret_tensor(primals_3, (256, 256), (256, 1), 0), reinterpret_tensor(primals_1, (256, 256), (256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 128, 256), (32768, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('Speech2Text2ForCausalLM', benchmark_compiled_module)
