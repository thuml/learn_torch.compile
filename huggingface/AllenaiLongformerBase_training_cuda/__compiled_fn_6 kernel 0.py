
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


# kernel path: /tmp/torchinductor_youkaichao/qd/cqddvba23z64qxckdhl7vgygbvfo2lxrhvj55rnlo3rezbsauvef.py
# Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
# token_type_ids => full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/vp/cvppritxkr4gia3h7zm72zdw4dolpwfokfzim66uai4tq6g75x3f.py
# Source Nodes: [extended_attention_mask_3], Original ATen: [aten.slice]
# extended_attention_mask_3 => full_default_1
triton_poi_fused_slice_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vsusfmrpsjc6nezvs5rqo2unfxyhwl5tij3gue2drtka4qm45y.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvh2mqjuivkqhtteq22xcfp4xqevstdkgtxl2aaevaulaabye2n.py
# Source Nodes: [add, add_1, embeddings, embeddings_1, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne]
# add => add
# add_1 => add_1
# embeddings => add_2
# embeddings_1 => add_3, add_4, mul_2, mul_3, rsqrt, sub_1, var_mean
# incremental_indices => mul_1
# inputs_embeds => embedding
# long => convert_element_type_2
# mask => convert_element_type
# ne => ne
# position_embeddings => embedding_1
# token_type_embeddings => embedding_2
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7 + tmp3
    tmp9 = tmp2 + 50265
    tmp10 = tmp2 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp2)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp11 < 50265")
    tmp12 = tl.load(in_ptr1 + (r1 + (768*tmp11)), rmask & xmask, other=0.0)
    tmp13 = tmp8 + 4098
    tmp14 = tmp8 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp8)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 4098)) | ~xmask, "index out of bounds: 0 <= tmp15 < 4098")
    tmp16 = tl.load(in_ptr2 + (r1 + (768*tmp15)), rmask & xmask, other=0.0)
    tmp17 = tmp12 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 768, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tmp19 - tmp29
    tmp37 = 768.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-05
    tmp40 = tmp38 + tmp39
    tmp41 = tl.math.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp41 / tmp37
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp42, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp46, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp47, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (50265, 768), (768, 1))
    assert_size_stride(primals_2, (4098, 768), (768, 1))
    assert_size_stride(primals_3, (1, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 1024, grid=grid(1024), stream=stream0)
        buf1 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [extended_attention_mask_3], Original ATen: [aten.slice]
        triton_poi_fused_slice_1.run(buf1, 1024, grid=grid(1024), stream=stream0)
        buf2 = empty((1, 1024), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        triton_poi_fused__to_copy_cumsum_ne_2.run(primals_6, buf2, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf3 = aten.cumsum(buf2, 1)
        del buf2
        buf4 = buf3
        del buf3
        buf5 = buf4; del buf4  # reuse
        buf6 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf10 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf11 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf15 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1, embeddings, embeddings_1, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne]
        triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_3.run(buf5, primals_6, primals_1, primals_2, primals_3, primals_4, primals_5, buf6, buf10, buf11, buf15, 1024, 768, grid=grid(1024), stream=stream0)
        del buf6
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        # Source Nodes: [embedding_output, embeddings_1], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf12 = aten.native_dropout(buf11, 0.1, True)
        del buf11
        buf13 = buf12[0]
        buf14 = buf12[1]
        return (buf13, buf1, primals_4, primals_6, buf0, buf5, buf10, buf14, buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4098, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
