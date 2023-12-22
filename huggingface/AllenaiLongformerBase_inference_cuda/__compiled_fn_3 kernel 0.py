
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


# kernel path: /tmp/torchinductor_youkaichao/we/cwempqvh5pk4bq6b7e7llk5f6qtlrxeks2ariqohja6tky2ieky3.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7x4j7pqmakw7o3a5qb5kghri4yl4b2ssupurdv6aiqi72bhj4c.py
# Source Nodes: [add, add_1, embeddings, embeddings_1, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, token_type_embeddings, token_type_ids, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.ne, aten.zeros]
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
# token_type_ids => full_default
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 50265
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp0 != tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp12 + 4098
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 4098)) | ~xmask, "index out of bounds: 0 <= tmp15 < 4098")
    tmp16 = tl.load(in_ptr3 + (r1 + (768*tmp15)), rmask & xmask, other=0.0)
    tmp17 = tmp4 + tmp16
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
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp46, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmon3nurx6pjfowutgpte23dslbc5rs73buvp4vehoyh24wbsogj.py
# Source Nodes: [extended_attention_mask_3], Original ATen: [aten.slice]
# extended_attention_mask_3 => full_default_1
triton_poi_fused_slice_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_2', 'mutated_arg_names': []},
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50265, 768), (768, 1))
    assert_size_stride(arg1_1, (4098, 768), (768, 1))
    assert_size_stride(arg2_1, (1, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(arg5_1, buf0, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1, embeddings, embeddings_1, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, token_type_embeddings, token_type_ids, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.ne, aten.zeros]
        triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1.run(arg5_1, arg0_1, buf2, arg1_1, arg2_1, arg3_1, arg4_1, buf3, buf7, 1024, 768, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf2
        del buf3
        buf8 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [extended_attention_mask_3], Original ATen: [aten.slice]
        triton_poi_fused_slice_2.run(buf8, 1024, grid=grid(1024), stream=stream0)
        return (buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4098, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
